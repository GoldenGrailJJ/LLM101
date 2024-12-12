import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from utils import generate

# Load the pre-trained GPT-2 model and tokenizer
model_name = "./models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)  # Load the model

# Set the padding token to be the same as the EOS token
# This ensures that the padding token and the end-of-sequence token are treated the same
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Set padding and truncation to occur on the left side of the input sequences
# This is a typical configuration for causal language models
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# Custom method to get dtype as float32 for GPT2Model (specific configuration for the model)
def get_float32_dtype(self):
    return torch.float32
GPT2Model.dtype = property(get_float32_dtype)  # Set the dtype property of GPT2Model

# Check the memory footprint of the model
model.get_memory_footprint()  # The model memory footprint is around 0.5 GB

# Define the quantization function
# Quantization scales the model's parameters to reduce their bit-width while maintaining reasonable accuracy
def quantize(t):
    min_val, max_val = t.min(), t.max()

    # Calculate the scale and zero_point for quantization
    scale = (max_val - min_val) / 255
    zero_point = min_val

    # Apply quantization: scale and shift values, clamp to the range [0, 255]
    t_quant = (t - zero_point) / scale
    t_quant = torch.clamp(t_quant, min=0, max=255)

    # Save the scaling parameters (state) for later use in dequantization
    state = (scale, zero_point)

    # Convert to unsigned 8-bit integer (quantized form)
    t_quant = t_quant.type(torch.uint8)
    return t_quant, state

# Define the dequantization function
# Dequantization converts the quantized tensor back to its original floating-point form
def dequantize(t, state):
    scale, zero_point = state
    # Perform the reverse operation of quantization: multiply by the scale and add zero_point
    return t.to(torch.float32) * scale + zero_point

# Define the function to quantize the entire model's parameters
def quantize_model(model):
    states = {}  # Dictionary to store the scaling factors (states) for each parameter
    for name, param in model.named_parameters():
        # Skip the gradient update for this step
        param.requires_grad = False
        # Quantize the parameter data and store the scaling state
        param.data, state = quantize(param.data)
        states[name] = state  # Save the state for the current parameter
    return model, states  # Return the quantized model and states

# Apply quantization to the model and save the states
quant_model, states = quantize_model(model)

# Define a helper function to compute the size of a tensor in bytes
def size_in_bytes(t):
    return t.numel() * t.element_size()  # numel() gives the number of elements, element_size() gives bytes per element

# Calculate the total size of all the quantization states (both scale and zero-point)
# The states dictionary holds both the scale and zero_point for each parameter, so we sum their sizes
total_size = sum([
    size_in_bytes(v[0]) + size_in_bytes(v[1])  # size of the quantized tensor and its state
    for v in states.values()  # Iterate over the quantization states
])
print(f"Total size of quantization states: {total_size} bytes")

# Function to dequantize the model by applying the stored scaling factors (states)
def dequantize_model(model, states):
    for name, param in model.named_parameters():
        state = states[name]  # Get the stored state for the current parameter
        # Dequantize the parameter using its respective scaling factor
        param.data = dequantize(param.data, state)
    return model  # Return the dequantized model

# Dequantize the model using the stored states
dequant_model = dequantize_model(quant_model, states)

# Use the `generate` function from the utils module to generate a response
# We provide an example prompt ("The quick brown fox jumped over the") and specify the number of tokens to generate (10)
response_expected = generate(
    dequant_model,
    tokenizer,
    [("The quick brown fox jumped over the", 10)]  # Example prompt with expected token length
)[0]

# Output the generated response
print(response_expected)
