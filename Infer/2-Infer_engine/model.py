import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Define a data class to hold model hyperparameters
@dataclass
class ModelArgs:
    # Default hyperparameters for the Llama 7B model
    dim: int = 4096  # Dimension of the model (hidden size)
    n_layers: int = 32  # Number of Transformer layers
    n_heads: int = 32  # Number of attention heads
    n_kv_heads: Optional[int] = None  # Number of key/value heads (if different from n_heads)
    vocab_size: int = 32000  # Vocabulary size for token embeddings
    hidden_dim: Optional[int] = None  # Dimension of the feedforward network (if specified)
    multiple_of: int = 256  # MLP hidden layer size will be a multiple of this value
    norm_eps: float = 1e-5  # Epsilon value for normalization layers to prevent division by zero
    max_seq_len: int = 2048  # Maximum sequence length for inputs
    dropout: float = 0.0  # Dropout rate for regularization


# Define a custom RMSNorm layer, a type of normalization layer
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps  # Epsilon to prevent division by zero
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter

    def _norm(self, x):
        """
        Compute the RMS (Root Mean Square) normalization.
            dim -1 means the dim of features, e.g. 4096
            weights should has the same dim as dim -1
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass for RMSNorm.
        """
        # Normalize the input tensor and apply the scaling parameter
        
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# 预计算用于旋转位置编码（Rotary Position Embedding, RoPE）的正弦和余弦频率成分
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码中所需的正弦（sine）和余弦（cosine）频率成分。

    参数:
        dim (int): 嵌入维度大小。
        end (int): 序列长度，即需要计算频率的位置数量。
        theta (float): 控制频率范围的缩放因子（默认值为 10000.0）。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 
        - 预计算的余弦频率张量，形状为 (end, dim//2)。
        - 预计算的正弦频率张量，形状为 (end, dim//2)。
    """
    # 第一步：计算每个嵌入维度的频率值
    # 公式为 1 / theta^(2i / dim)，其中 i 是偶数索引
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 第二步：生成位置索引 (0, 1, ..., end - 1)
    t = torch.arange(end, device=freqs.device)

    # 第三步：计算位置索引与频率的外积
    # 得到每个位置与维度对应的频率值
    freqs = torch.outer(t, freqs).float()

    # 第四步：计算频率值的正弦和余弦
    freqs_cos = torch.cos(freqs)  # 余弦值（对应复数的实部）
    freqs_sin = torch.sin(freqs)  # 正弦值（对应复数的虚部）

    return freqs_cos, freqs_sin

# 调整频率张量形状，使其可与输入张量进行广播运算
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    将频率张量调整为可与输入张量进行广播的形状。

    参数:
        freqs_cis (torch.Tensor): 预计算的余弦或正弦频率张量，形状为 (seq_len, dim//2)。
        x (torch.Tensor): 输入张量，形状为 (..., seq_len, dim)。

    返回:
        torch.Tensor: 调整后的频率张量，与输入张量的形状兼容。
    """
    # 获取输入张量的维度数量
    ndim = x.ndim
    # 确保输入张量的维度至少有序列长度和嵌入维度
    assert 0 <= 1 < ndim
    # 确保频率张量的形状与输入张量的序列长度和最后一个维度相匹配
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 构造与输入张量兼容的形状
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    # 调整频率张量的形状
    return freqs_cis.view(shape)

# 将旋转位置编码（RoPE）应用到查询（Query）和键（Key）张量上
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对查询张量 (xq) 和键张量 (xk) 应用旋转位置编码 (RoPE)。

    参数:
        xq (torch.Tensor): 查询张量，形状为 (..., seq_len, dim)。
        xk (torch.Tensor): 键张量，形状为 (..., seq_len, dim)。
        freqs_cos (torch.Tensor): 预计算的余弦频率张量。
        freqs_sin (torch.Tensor): 预计算的正弦频率张量。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 
        - 应用 RoPE 后的查询张量 (xq_out)。
        - 应用 RoPE 后的键张量 (xk_out)。
    """
    # 将查询张量和键张量最后一个维度拆分为实部和虚部，形状为 (..., seq_len, dim//2, 2)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 将频率张量调整为与实部和虚部广播兼容的形状
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 对查询张量的实部和虚部应用旋转公式
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # 实部
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # 虚部

    # 对键张量的实部和虚部应用旋转公式
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # 实部
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # 虚部

    # 将实部和虚部重新组合并展平回原始形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

