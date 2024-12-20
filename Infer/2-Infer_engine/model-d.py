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
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass for RMSNorm.
        """
        # Normalize the input tensor and apply the scaling parameter
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the cosine and sine frequencies for rotary positional embeddings.

    Args:
        dim (int): Dimension of the embeddings.
        end (int): The maximum sequence length.
        theta (float): A scaling factor for frequencies.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine frequencies.
    """
    # Compute the frequency ratios
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # Create a range from 0 to end-1
    freqs = torch.outer(t, freqs).float()  # Compute the outer product to get frequency matrix
    freqs_cos = torch.cos(freqs)  # Real part of the frequencies
    freqs_sin = torch.sin(freqs)  # Imaginary part of the frequencies
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape the frequency tensors to enable broadcasting with input tensor x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensors (cosine or sine).
        x (torch.Tensor): Input tensor to which frequencies will be applied.

    Returns:
        torch.Tensor: Reshaped frequency tensor ready for broadcasting.
    """
    ndim = x.ndim  # Number of dimensions in x
    assert 0 <= 1 < ndim  # Ensure that the second dimension exists
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # Check shape compatibility
    # Create a shape list where only the second and last dimensions match x's shape
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)  # Reshape freqs_cis for broadcasting


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Positional Embeddings (RoPE) to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        freqs_cos (torch.Tensor): Precomputed cosine frequencies.
        freqs_sin (torch.Tensor): Precomputed sine frequencies.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    # Reshape xq and xk to separate real and imaginary parts for complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape frequencies for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply rotation using real number arithmetic
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # Real part after rotation
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # Imaginary part after rotation
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # Real part after rotation
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # Imaginary part after rotation

    # Stack and flatten the rotated real and imaginary parts back into single tensors
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat the key/value tensors along the head dimension.

    Args:
        x (torch.Tensor): Key or value tensor.
        n_rep (int): Number of repetitions.

    Returns:
        torch.Tensor: Repeated key/value tensor.
    """
    # Extract the shape components
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x  # No repetition needed
    # Expand and reshape the tensor to repeat the key/value heads
    return (
        x[:, :, :, None, :]  # Add a new dimension for repetition
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # Expand along the new dimension
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # Merge the repeated heads
    )


# Define the Attention module
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Determine the number of key/value heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        # Model parallelism settings (currently set to 1)
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size  # Local attention heads
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size  # Local key/value heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # Number of repetitions for key/value heads

        # Dimension per attention head
        self.head_dim = args.dim // args.n_heads

        # Define linear projections for queries, keys, and values
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # Query projection
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Key projection
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Value projection
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # Output projection

        # Define dropout layers for attention weights and residual connections
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # Check if Flash Attention is available (requires PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Create an upper triangular mask for causal (autoregressive) attention
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)  # Only allow attention to previous tokens
            self.register_buffer("mask", mask)  # Register mask as a non-trainable buffer

    def forward(
            self,
            x: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
    ):
        """
        Forward pass for the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
            freqs_cos (torch.Tensor): Precomputed cosine frequencies for RoPE.
            freqs_sin (torch.Tensor): Precomputed sine frequencies for RoPE.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        bsz, seqlen, _ = x.shape  # Batch size and sequence length

        # Project input to queries, keys, and values
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Reshape queries to (batch_size, sequence_length, n_local_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # Reshape keys and values to (batch_size, sequence_length, n_local_kv_heads, head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply Rotary Positional Embeddings to queries and keys
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Repeat key/value heads if necessary for grouped multi-query attention
        xk = repeat_kv(xk, self.n_rep)  # Shape: (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # Shape: (bs, seqlen, n_local_heads, head_dim)

        # Transpose to bring heads to the batch dimension for efficient computation
        xq = xq.transpose(1, 2)  # Shape: (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # Shape: (bs, n_local_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # Shape: (bs, n_local_heads, seqlen, head_dim)

        if self.flash:
            # Use Flash Attention if available for optimized performance
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # Ensure causal (autoregressive) attention
            )
        else:
            # Manual implementation of scaled dot-product attention
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)  # Compute attention scores
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # Apply causal mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # Softmax over the last dimension
            scores = self.attn_dropout(scores)  # Apply dropout to attention weights
            output = torch.matmul(scores, xv)  # Weighted sum of values

        # Transpose back and reshape to (batch_size, sequence_length, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Apply the output projection and residual dropout
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


# Define the FeedForward module (MLP)
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            # If hidden_dim is not specified, compute it based on multiple_of
            hidden_dim = 4 * dim  # Typically, hidden_dim is 4 times the input dim
            hidden_dim = int(2 * hidden_dim / 3)  # Adjust hidden_dim
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # Ensure multiple_of
        # Define linear layers without bias for efficiency
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # First linear layer
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Second linear layer
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Third linear layer (element-wise gating)
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x):
        """
        Forward pass for the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor after feedforward computation.
        """
        # Apply first linear layer and activation
        activated = F.silu(self.w1(x))  # Apply SiLU activation
        # Apply gating with the third linear layer
        gated = activated * self.w3(x)
        # Apply second linear layer and dropout
        return self.dropout(self.w2(gated))


# Define a single Transformer block consisting of Attention and FeedForward modules
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads  # Number of attention heads
        self.dim = args.dim  # Hidden dimension
        self.head_dim = args.dim // args.n_heads  # Dimension per head
        self.attention = Attention(args)  # Attention module
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )  # FeedForward module
        self.layer_id = layer_id  # Identifier for the layer
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # Normalization before attention
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # Normalization before feedforward

    def forward(self, x, freqs_cos, freqs_sin):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
            freqs_cos (torch.Tensor): Precomputed cosine frequencies for RoPE.
            freqs_sin (torch.Tensor): Precomputed sine frequencies for RoPE.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        # Apply normalization and attention, then add residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        # Apply normalization and feedforward, then add residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# Define the complete Transformer model
class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]  # Attribute to store the loss from the last forward pass

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params  # Store model hyperparameters
        self.vocab_size = params.vocab_size  # Vocabulary size
        self.n_layers = params.n_layers  # Number of Transformer layers

        # Define token embeddings
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)  # Dropout layer after embeddings

        # Create a stack of Transformer blocks
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # Define normalization layer after the Transformer blocks
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Define the output projection layer
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Share the output projection weights with the token embeddings (weight tying)
        self.tok_embeddings.weight = self.output.weight  # This ties the input and output embeddings

        # Precompute frequencies for Rotary Positional Embeddings (RoPE)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # Register as non-trainable buffers
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize all weights in the model
        self.apply(self._init_weights)
        # Apply special scaled initialization to residual projections as per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call
        self.last_loss = None

    def _init_weights(self, module):
        """
        Initialize weights of the model.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Initialize biases to zero
        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            targets (Optional[torch.Tensor]): Target token IDs for computing loss.

        Returns:
            torch.Tensor: Output logits for the next token prediction.
        """
        _bsz, seqlen = tokens.shape  # Batch size and sequence length

        # Embed the input tokens
        h = self.tok_embeddings(tokens)  # Shape: (batch_size, sequence_length, dim)
        h = self.dropout(h)  # Apply dropout to embeddings

        # Slice the precomputed frequencies up to the current sequence length
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Pass through each Transformer layer
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # Apply final normalization
        h = self.norm(h)

        if targets is not None:
            # If targets are provided, compute the loss
            logits = self.output(h)  # Compute logits for all tokens
            # Compute cross-entropy loss, ignoring padding tokens (-1 index)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference mode: only compute logits for the last token
            logits = self.output(h[:, [-1], :])  # Shape: (batch_size, 1, vocab_size)
            self.last_loss = None  # No loss in inference mode

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure and return the optimizer for training.

        Args:
            weight_decay (float): Weight decay (L2 regularization) factor.
            learning_rate (float): Learning rate for the optimizer.
            betas (Tuple[float, float]): Beta coefficients for the Adam optimizer.
            device_type (str): Type of device ('cuda' or 'cpu').

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        # Create a dictionary of all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters into those that will have weight decay and those that won't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # Typically weights
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # Typically biases and norms

        # Define optimizer parameter groups with appropriate weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Count parameters for logging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Check if fused AdamW is available (requires specific PyTorch versions and CUDA)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # Instantiate the AdamW optimizer with the parameter groups
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate the Model FLOPs Utilization (MFU) in terms of A100 bfloat16 peak FLOPS.

        Args:
            fwdbwd_per_iter (int): Number of forward-backward passes per iteration.
            dt (float): Time per iteration in seconds.

        Returns:
            float: Estimated MFU as a ratio of A100's peak FLOPS.
        """
        # Estimate the total number of floating-point operations per token
        N = sum(p.numel() for p in self.parameters())  # Total number of parameters
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T  # Total FLOPs per token
        flops_per_fwdbwd = flops_per_token * T  # FLOPs per forward-backward pass
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter  # Total FLOPs per iteration

        # Calculate FLOPs achieved per second
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPs is 312 TFLOPS

        # Compute MFU as the ratio of achieved FLOPs to A100's peak FLOPs
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens based on input indices using the trained model.

        Args:
            idx (torch.Tensor): Tensor of input token IDs of shape (batch_size, sequence_length).
            max_new_tokens (int): Number of new tokens to generate.
            temperature (float): Sampling temperature; higher values increase randomness.
            top_k (Optional[int]): If set, limits sampling to the top_k most probable tokens.

        Returns:
            torch.Tensor: Tensor of generated token IDs.
        """
        for _ in range(max_new_tokens):
            # If the context is too long, truncate it to the maximum sequence length
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # Forward pass to get logits for the current sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on the logits of the last token

            if temperature == 0.0:
                # Deterministic sampling: choose the token with the highest probability
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # Apply temperature scaling to logits
                logits = logits / temperature
                if top_k is not None:
                    # If top_k is specified, filter logits to keep only the top_k tokens
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')  # Mask out tokens outside top_k
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)
                # Sample the next token from the probability distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # Append the sampled token to the input sequence for the next iteration
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
