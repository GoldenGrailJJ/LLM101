import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Set the seed to ensure reproducible results for each run
torch.manual_seed(42)

# Define a simple test model with an embedding layer, a linear layer, and a language modeling head
class TestModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Embedding layer to convert input tokens to embeddings of size 'hidden_size'
        self.embedding = torch.nn.Embedding(10, hidden_size)
        # Linear layer to project the embeddings into another space of the same size (hidden_size)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        # Language modeling head to map the hidden states to the token space
        self.lm_head = torch.nn.Linear(hidden_size, 10)
    
    def forward(self, input_ids):
        # Pass the input_ids through the embedding layer
        x = self.embedding(input_ids)
        # Apply a linear transformation to the embeddings
        x = self.linear(x)
        # Pass the result through the language modeling head to generate logits
        x = self.lm_head(x)
        return x  # Return logits for each token in the sequence

# Define the hidden size for the model
hidden_size = 1024
# Instantiate the test model with the specified hidden size
model = TestModel(hidden_size)

# Example input ids (a sequence of token ids)
input_ids = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])

# Define a simple detokenizer that maps token ids to color names
detokenizer = [
    "red",        # Token ID 0
    "orange",     # Token ID 1
    "yellow",     # Token ID 2
    "green",      # Token ID 3
    "blue",       # Token ID 4
    "indigo",     # Token ID 5
    "violet",     # Token ID 6
    "magenta",    # Token ID 7
    "marigold",   # Token ID 8
    "chartreuse", # Token ID 9
]

# Generate the next token based on the model's prediction
def generate_token(model, input_ids):
    with torch.no_grad():
        # Perform a forward pass to get the logits for each token
        logits = model(input_ids)  
    # Extract the logits corresponding to the last token in the sequence
    last_logits = logits[:, -1, :]  
    # Get the token ID with the highest logit value (i.e., most likely token)
    next_token_ids = last_logits.argmax(dim=1)  
    # Convert the token IDs to words using the detokenizer
    return [detokenizer[token_id.item()] for token_id in next_token_ids]

# Generate the next token for the input sequence
next_token = generate_token(model, input_ids)[0]
# Print the generated token
print(next_token)

# Example of Low-Rank Adaptation (LoRA) applied to a linear layer
X = torch.randn(1, 8, 1024)  # Input tensor with shape (batch_size, seq_len, hidden_size)

# Create low-rank matrices for LoRA adaptation
lora_a = torch.randn(1024, 2)  # Low-rank matrix A (hidden_size, r)
lora_b = torch.randn(2, 1024)  # Low-rank matrix B (r, hidden_size)

# Base output from the linear layer without LoRA
base_output = model.linear(X)
# Output from the LoRA mechanism (rank-r adaptation)
lora_output = X @ lora_a @ lora_b

# Combine the outputs of the linear layer and LoRA (additive fusion)
total_output = base_output + lora_output

# Define a class that implements a LoRA layer for neural network layers
class LoraLayer(torch.nn.Module):
    def __init__(self, base_layer, r):
        super().__init__()
        # Store the base layer (e.g., linear layer) and rank 'r' for LoRA
        self.base_layer = base_layer

        # Get input and output dimensions of the base layer
        d_in, d_out = self.base_layer.weight.shape

        # Create low-rank matrices A and B (with rank 'r')
        self.lora_a = torch.randn(d_in, r)  # Low-rank matrix A (input_dim, r)
        self.lora_b = torch.randn(r, d_out)  # Low-rank matrix B (r, output_dim)

    def forward(self, x):
        # Apply the base layer to the input
        y1 = self.base_layer(x)  
        # Apply LoRA adaptation (matrix multiplication of input with A and B)
        y2 = x @ self.lora_a @ self.lora_b  
        # Return the sum of the base output and the LoRA adaptation
        return y1 + y2  

# Apply LoRA to the linear layer of the model
lora_layer = LoraLayer(model.linear, 2)

# Run the input tensor through the LoRA-augmented layer
output = lora_layer(X)
# Print the output shape
print(output.shape)  # Expected shape: (1, 8, 1024)
