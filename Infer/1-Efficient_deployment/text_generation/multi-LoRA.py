import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Abstract model class for multi-LoRA (Low-Rank Adaptation) models
class AbstractMultiLoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Basic layers for the model: embedding, linear layer, and LM head
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)
    
    def linear_lora(
        self,
        x: torch.Tensor,                 # (batch_size, seq_len, in_features)
        loras_a: torch.Tensor,           # (num_loras, in_features, rank)
        loras_b: torch.Tensor,           # (num_loras, rank, out_features)
        lora_indices: torch.LongTensor,  # (batch_size,)
    ) -> torch.Tensor:
        """
        Placeholder method for linear LoRA application.
        This will be implemented in subclasses.
        """
        raise NotImplementedError()

    def forward(self, input_ids, loras_a, loras_b, lora_indices):
        """
        Forward pass of the model. The input goes through embedding,
        the LoRA weights are applied, and then through the LM head.
        """
        x = self.embedding(input_ids)  # Apply embedding to input_ids
        x = self.linear_lora(x, loras_a, loras_b, lora_indices)  # Apply LoRA
        x = self.lm_head(x)  # Apply language model head
        return x


# A subclass implementation for applying LoRA using a loop
class LoopMultiLoraModel(AbstractMultiLoraModel):
    def linear_lora(
        self,
        x: torch.Tensor,                 # (batch_size, seq_len, in_features)
        loras_a: torch.Tensor,           # (num_loras, in_features, lora_rank)
        loras_b: torch.Tensor,           # (num_loras, lora_rank, out_features)
        lora_indices: torch.LongTensor,  # (batch_size,)
    ) -> torch.Tensor:
        """
        Apply LoRA weights to the input tensor using a loop over batch indices.
        """
        y = self.linear(x)  # Apply a linear transformation
        
        # Loop over the batch and apply the LoRA weights for each sample
        for batch_idx, lora_idx in enumerate(lora_indices.numpy()):
            lora_a = loras_a[lora_idx]  # Select the LoRA matrix A for this batch item
            lora_b = loras_b[lora_idx]  # Select the LoRA matrix B for this batch item
            y[batch_idx] += x[batch_idx] @ lora_a @ lora_b  # Apply LoRA modification

        return y


# A toy detokenizer example to map token IDs to readable words
detokenizer = [
    "red", "orange", "yellow", "green", "blue", 
    "indigo", "violet", "magenta", "marigold", "chartreuse",
]

# Function to generate a token based on the model's output logits
def generate_token(model, **kwargs):
    with torch.no_grad():
        logits = model(**kwargs)  # Get the logits from the model
    last_logits = logits[:, -1, :]  # Focus on the last token
    next_token_ids = last_logits.argmax(dim=1)  # Get the token with the highest probability

    # Convert token IDs to words using the detokenizer
    return [detokenizer[token_id] for token_id in next_token_ids]


# Constants for the model and benchmark tests
bs = 1
num_loras = 64  # Number of LoRA weight sets
h = 10  # Feature dimension
r = 2  # Rank of the LoRA

# Random initialization of LoRA matrices
loras_a = torch.randn(num_loras, h, r)
loras_b = torch.randn(num_loras, r, h)

# Generate tokens for a batch of 10 samples
for i in range(10):
    lora_indices = torch.randint(num_loras, (bs,), dtype=torch.long)
    next_token = generate_token(
        model,
        input_ids=input_ids,
        lora_a=loras_a,
        lora_b=loras_b,
        lora_indices=lora_indices
    )


# Constants for the benchmark test
seq_len = 8
vocab_size = 10
nsamples = 500
max_batch_size = 64

# Benchmark function to measure latency for different batch sizes
def benchmark(model):
    avg_latencies = []
    for bs in range(1, max_batch_size + 1):  # Loop through different batch sizes
        latencies = []
        for _ in range(nsamples):  # Repeat for multiple samples
            # Randomize the inputs and LoRA indices for each sample
            input_ids = torch.randint(vocab_size, (bs, seq_len), dtype=torch.long)
            lora_indices = torch.randint(num_loras, (bs,), dtype=torch.long)

            # Measure the latency of generating a single token
            t0 = time.time()
            next_token = generate_token(
                model,
                input_ids=input_ids,
                loras_a=loras_a,
                loras_b=loras_b,
                lora_indices=lora_indices,
            )
            latencies.append(time.time() - t0)  # Record the time taken

        # Average the latency across all samples
        latency_s = sum(latencies) / len(latencies)
        avg_latencies.append(latency_s)
        print(bs, latency_s)  # Print the batch size and average latency
    return avg_latencies


# A subclass implementation for applying LoRA using tensor gathering
class GatheredMultiLoraModel(AbstractMultiLoraModel):
    def linear_lora(
        self,
        x: torch.Tensor,                 # (batch_size, seq_len, in_features)
        loras_a: torch.Tensor,           # (num_loras, in_features, lora_rank)
        loras_b: torch.Tensor,           # (num_loras, lora_rank, out_features)
        lora_indices: torch.LongTensor,  # (batch_size,)
    ) -> torch.Tensor:
        """
        Gather LoRA weights based on the indices and apply them to the input tensor.
        """
        y = self.linear(x)  # Apply the initial linear transformation
        
        # Gather the LoRA weight matrices based on the indices
        lora_a = torch.index_select(loras_a, 0, lora_indices)  # (batch_size, in_features, lora_rank)
        lora_b = torch.index_select(loras_b, 0, lora_indices)  # (batch_size, lora_rank, out_features)

        # Apply LoRA modification
        y += x @ lora_a @ lora_b
        return y
