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

