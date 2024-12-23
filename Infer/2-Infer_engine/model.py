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

# 重复键和值的头部
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键或值张量的头部。
    
    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)。
        n_rep (int): 每个头需要重复的次数。
    
    返回:
        torch.Tensor: 输出张量，形状为 (batch_size, seq_len, n_kv_heads * n_rep, head_dim)。
    """
    # 获取输入张量的形状
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # 如果重复次数为 1，直接返回原张量
    if n_rep == 1:
        return x
    
    # 扩展并重复键和值的头部
    return (
        x[:, :, :, None, :]  # 新增一个维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展到 n_rep 的重复次数
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 调整形状为目标输出
    )

# 注意力机制的实现
class Attention(nn.Module):
    def __init__(self, args):
        """
        初始化 Attention 模块。
        
        参数:
            args: 模型参数，包括头数量、嵌入维度、丢弃率等。
        """
        super().__init__()
        # 键值头的数量，如果未指定则等于总头数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        # 确保总头数可以被键值头数量整除
        assert self.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size  
        # 计算重复次数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.dim // self.n_heads
        
        # ? 定义线性层，用于生成 Q, K, V
        self.wq = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.fill((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        """
        前向传播，计算注意力结果。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。
            freqs_cos (torch.Tensor): RoPE 编码的余弦分量。
            freqs_sin (torch.Tensor): RoPE 编码的正弦分量。
        
        返回:
            torch.Tensor: 注意力输出。
        """
        bcz, seqlen, _ = x.shape()

        # Step 1: 生成 Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # 使用线性层生成
        xq = xq.view(bcz, seqlen, self.n_local_heads, self.head_dim)  # 调整 Q 的形状
        xk = xk.view(bcz, seqlen, self.n_local_kv_heads, self.head_dim)  # 调整 K 的形状
        xv = xv.view(bcz, seqlen, self.n_local_kv_heads, self.head_dim)  # 调整 V 的形状

        # Step 2: 应用 RoPE 编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)  # 调用 apply_rotary_emb

        # Step 3: 重复键和值的头部
        xk = repeat_kv(xk, self.n_rep)  # 重复键
        xv = repeat_kv(xv, self.n_rep)  # 重复值

        # Step 4: 转置维度，为注意力计算做准备
        xq = xq.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Step 5: 使用 Flash Attention 或传统实现
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim))
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores, dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv) # (bs, n_local_heads, seqlen, head_dim)

        # Step 6: 恢复张量的原始形状
        output = output.transpose(1, 2).contiguous().view(bcz, seqlen, -1)

        # Step 7: 最终的线性投影
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果未提供 hidden_dim，则自动计算
        if hidden_dim is None:
            hidden_dim = 4 * dim  # 默认为 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)  # 缩减为 2/3
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 向上取整为 multiple_of 的倍数
        # 定义线性层
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义 Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 计算 FeedForward 输出
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        # 初始化参数
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads  # 每个头的维度

        # 初始化注意力机制
        self.attention = Attention(args)

        # 初始化前馈网络
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        # 层归一化
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 通过注意力模块
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        # 通过前馈网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]  # 用于存储最近一次前向传播的损失

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size  # 词汇表大小
        self.n_layers = params.n_layers  # Transformer 层的数量

        # 定义嵌入层和 Dropout
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  # 词向量嵌入层
        self.dropout = nn.Dropout(params.dropout)  # 随机丢弃层

        # 定义 Transformer 层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))  # 添加每一层 TransformerBlock

        # 定义归一化层和输出层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # RMSNorm 用于稳定训练
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 输出层将维度映射回词汇表大小

        # 权重共享（权重绑定）
        self.tok_embeddings.weight = self.output.weight  # 嵌入层和输出层共享权重

        # 预计算 RoPE 相对位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len  # 计算正弦和余弦频率
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 注册正弦编码
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # 注册余弦编码

        # 初始化所有权重
        self.apply(self.init_weights)

        # 特殊初始化，用于残差投影层（参考 GPT-2 论文）
        for pn, p in self.named_parameters():
            """
            遍历模型的所有可训练参数。
            - `pn`: 参数的名称（字符串），例如 'layer.0.attention.w3.weight'。
            - `p`: 参数的实际张量（Tensor）。
            """
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                """
                判断参数名称是否以 'w3.weight' 或 'wo.weight' 结尾。
                - `w3.weight`: 通常是前馈网络（FeedForward）的第三个线性层的权重。
                - `wo.weight`: 通常是注意力模块的输出线性层的权重。
                - 这两类权重在 Transformer 中与残差连接的输出直接相关，因此需要特殊初始化。
                """
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * params.n_layers)
                )
                """
                对这些参数进行正态分布初始化：
                - `mean=0.0`：均值为 0。
                - `std=0.02 / math.sqrt(2 * params.n_layers)`：标准差根据 GPT-2 初始化规则计算。
                    - `0.02` 是 GPT 系列默认的初始化缩放因子。
                    - `math.sqrt(2 * params.n_layers)` 通过层数缩放，防止随着层数增加，残差流中的梯度爆炸。
                """

        # 初始化记录最近一次前向传播损失的变量
        self.last_loss = None
    
    def _init_weights(self, module):
        """
        初始化模型权重：
        - Linear 层使用正态分布初始化权重，均值为 0，标准差为 0.02。
        - Embedding 层使用相同的正态分布初始化权重。
        - 如果有 bias，则初始化为 0。
        """
        if isinstance(module, nn.Linear):  # 判断是否是线性层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化权重
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 初始化 bias
        elif isinstance(module, nn.Embedding):  # 判断是否是嵌入层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化权重

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        实现前向传播：
        - tokens: 输入的词索引张量，形状为 (batch_size, seq_len)。
        - targets: 目标词索引张量，形状为 (batch_size, seq_len)。
        """
        _bsz, seqlen = tokens.shape  # 获取 tokens 的形状

        # 步骤 1: 嵌入输入的词索引并添加 Dropout
        h = self.tok_embeddings(tokens)  # 使用嵌入层将 tokens 转换为嵌入表示
        h = self.dropout(h)  # 添加随机丢弃

        # 步骤 2: 提取对应序列长度的 RoPE 编码
        freqs_cos = self.freqs_cos[:seqlen]  # 获取 RoPE 的正弦部分
        freqs_sin = self.freqs_sin[:seqlen]  # 获取 RoPE 的余弦部分

        # 步骤 3: 依次通过每个 Transformer 层
        for layer in self.layers:  # 遍历每个 Transformer 层
            h = layer(h, freqs_cos, freqs_sin)  # 输入当前层并更新 h

        # 步骤 4: 最后一层归一化
        h = self.norm(h)  # 通过归一化层

        # 步骤 5: 输出 logits 或计算损失
        if targets is not None:
            logits = self.output(h)  # 输出 logits
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # 展平 logits
                targets.view(-1),  # 展平目标
                ignore_index=-1  # 忽略填充位置
            )
        else:
            logits = self.output(h[:, [-1], :])  # 仅输出最后一个时间步
            self.last_loss = None

        return logits
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        配置优化器函数，使用分组参数以优化模型的训练性能。

        参数：
        - weight_decay: 权重衰减系数，用于正则化（L2 范数）。
        - learning_rate: 学习率，控制参数更新的步长。
        - betas: AdamW 优化器的超参数 (beta1, beta2)，用于一阶和二阶动量的计算。
        - device_type: 设备类型，通常为 'cuda' 或 'cpu'，决定是否启用 fused 优化器。

        返回值：
        - optimizer: 配置好的 AdamW 优化器实例。
        """
        # 第一步：获取所有模型参数
        # `self.named_parameters()` 返回模型中所有参数及其名称
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # 第二步：过滤掉不需要梯度计算的参数
        # 只有 `requires_grad=True` 的参数会参与优化，通常冻结层的参数会被过滤掉
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 第三步：根据参数维度将其分为两组
        # 1. 需要应用权重衰减的参数（`dim >= 2` 的张量，例如权重矩阵和嵌入矩阵）
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]

        # 2. 不需要权重衰减的参数（`dim < 2` 的张量，例如偏置项和 LayerNorm 参数）
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # 第四步：定义优化器的参数组
        # 权重衰减组和无权重衰减组分别配置
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},  # 应用权重衰减
            {'params': nodecay_params, 'weight_decay': 0.0}          # 无权重衰减
        ]

        # 打印参数信息
        num_decay_params = sum(p.numel() for p in decay_params)  # 统计需要权重衰减的参数数量
        num_nodecay_params = sum(p.numel() for p in nodecay_params)  # 统计不需要权重衰减的参数数量
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 第五步：选择是否启用 fused 优化器
        # 检查 torch.optim.AdamW 是否支持 fused 参数
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # 如果 fused 可用且设备是 CUDA，则启用 fused 版本的 AdamW 优化器
        use_fused = fused_available and device_type == 'cuda'
        # 如果使用 fused 优化器，则添加对应参数
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        # 打印是否使用 fused 优化器
        print(f"using fused AdamW: {use_fused}")

        # 返回配置好的优化器
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估算模型的 FLOPS 利用率（MFU, Model FLOPS Utilization），单位为 A100 GPU bfloat16 的峰值 FLOPS。
        
        参数：
        - fwdbwd_per_iter: 每次迭代中前向和反向传播的次数，通常为 1 或 2。
        - dt: 每次迭代的耗时（以秒为单位）。
        
        返回值：
        - mfu: 模型 FLOPS 利用率，表示实际 FLOPS 与 A100 bfloat16 峰值 FLOPS 的比值。
        """
        # 第一步：估算每次迭代中执行的 FLOPS（浮点运算次数）。
        # 参考 PaLM 论文的 Appendix B：https://arxiv.org/abs/2204.02311

        # 1. 计算模型中的总参数数量 N
        N = sum(p.numel() for p in self.parameters())  # 所有参数的总数量（标量计数）

        # 2. 提取模型的配置参数
        cfg = self.params
        L = cfg.n_layers       # 模型的层数
        H = cfg.n_heads        # 注意力头的数量
        Q = cfg.dim // cfg.n_heads  # 每个注意力头的维度
        T = cfg.max_seq_len    # 最大序列长度

        # 3. 计算每个 token 的 FLOPS
        # 每个 token 的 FLOPS 包括：
        # - 6N：完全连接层的权重乘加运算，N 是参数总数
        # - 12*L*H*Q*T：注意力机制的计算量，涉及 Q、K、V 的点积和缩放操作
        flops_per_token = 6 * N + 12 * L * H * Q * T

        # 4. 计算每次前向和反向传播的 FLOPS
        # 每个前向+反向传播需要处理整个序列长度 T，因此：
        flops_per_fwdbwd = flops_per_token * T

        # 5. 计算每次迭代的 FLOPS
        # 每次迭代可能包括多个前向和反向传播（由 fwdbwd_per_iter 决定）。
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # 第二步：将实际 FLOPS 转化为每秒的 FLOPS
        flops_achieved = flops_per_iter * (1.0 / dt)  # 每秒实际执行的 FLOPS

        # 第三步：计算 A100 GPU 的峰值 bfloat16 FLOPS
        flops_promised = 312e12  # A100 GPU bfloat16 的峰值性能为 312 TFLOPS（312 * 10^12 FLOPS）

        # 第四步：计算模型 FLOPS 利用率
        mfu = flops_achieved / flops_promised  # 实际 FLOPS 与峰值 FLOPS 的比值

        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        实现文本生成逻辑：
        - idx: 初始输入序列 (LongTensor，形状为 (batch_size, seq_len))。
        - max_new_tokens: 生成的新词数量。
        - temperature: 采样温度，控制生成的多样性。
        - top_k: 限制采样时只考虑概率最高的前 k 个词（如果设置为 None，则不限制）。
        """
        for _ in range(max_new_tokens):  # 逐步生成 max_new_tokens 个新词
            # 1. 如果序列过长，裁剪到最大上下文长度
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            """
            如果长度小于或等于模型允许的最大序列长度（self.params.max_seq_len），直接使用整个序列。
            如果长度超过最大长度，则截断，只保留最后 max_seq_len 个词。
            """

            # 2. 前向传播，获取当前序列的 logits
            logits = self(idx_cond)  # 模型前向计算
            logits = logits[:, -1, :]  # 只取最后一个时间步的 logits
            """
            logits.shape: (batch_size, seq_len, vocab_size)。
            经过 [:, -1, :] 裁剪后，形状变为 (batch_size, vocab_size)，对应当前时间步的预测分布。
            """

            # 3. 根据温度采样或直接选择最高概率的词
            if temperature == 0.0:
                # 如果温度为 0，选择概率最高的词（贪心搜索）
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 对 logits 进行温度缩放
                logits = logits / temperature

                # 如果设置了 top_k，仅保留概率最高的前 k 个词
                if top_k is not None:
                    # values, indices = torch.topk(input, k, dim=-1, largest=True, sorted=True)
                    # input: 输入张量, k: 保留的最大元素数, dim: 沿着哪个维度计算
                    # values: 保留的最大元素值, indices: 保留的最大元素索引
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # v[:, [-1]] 表示每行的第k大的值
                    logits[logits < v[:, [-1]]] = -float('Inf')  # 将其他词的概率置为负无穷

                # 将 logits 转换为概率分布
                probs = F.softmax(logits, dim=-1)

                # 根据概率分布采样下一个词
                # torch.multinomial 函数用于从给定的概率分布中随机采样
                # 参数解释：
                #   probs: 形状为 (batch_size, vocab_size) 的二维张量，每行是一个概率分布
                #          每个概率分布是通过 softmax 得到的，表示当前词汇表中每个词的概率
                #   num_samples=1: 表示对每一行的概率分布采样 1 次，生成一个词的索引
                # 返回值：
                #   idx_next: 形状为 (batch_size, num_samples) 的二维张量，包含采样结果的索引
                #   - 每行的索引表示在对应的概率分布中采样得到的词的索引
                #   - 因为 num_samples=1，返回的每行只有一个采样索引
                idx_next = torch.multinomial(probs, num_samples=1)

            # 4. 将采样得到的新词追加到序列中
            idx = torch.cat((idx, idx_next), dim=1)
            """
            torch.cat: 将新词 idx_next 添加到原序列 idx 的末尾。
            idx.shape: 逐步增长，最终形状为 (batch_size, seq_len + max_new_tokens)。
            """

        # 5. 返回完整生成的序列
        return idx