import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load model and tokenizer
model_name = "./models/gpt2"  # Path to the pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Initialize tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)  # Load the pre-trained model

# Define the input prompt
prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")  # Tokenize the prompt and return tensors


# Function to generate a token given past key values (used for caching)
def generate_token_with_past(inputs):
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**inputs)  # Perform forward pass through the model

    logits = outputs.logits  # Get the logits (predicted scores for the next token)
    last_logits = logits[0, -1, :]  # Extract the logits for the last token of the sequence
    next_token_id = last_logits.argmax()  # Get the token ID with the highest score (argmax)
    return next_token_id, outputs.past_key_values  # Return the token ID and past key values for caching


# Function to generate text from an input prompt
def generate(inputs, max_tokens):
    generated_tokens = []  # List to store the generated tokens
    next_inputs = inputs  # Set the initial inputs to the original prompt

    for _ in range(max_tokens):  # Generate tokens up to the specified maximum
        next_token_id, past_key_values = generate_token_with_past(next_inputs)  # Get the next token ID and past key values

        # Update inputs for the next token generation
        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),  # Reshape to match input format (batch size, sequence length)
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]])],  # Extend attention mask by 1
                dim=1
            ),
            "past_key_values": past_key_values,  # Pass the cached key values for efficiency
        }

        next_token = tokenizer.decode(next_token_id)  # Decode the token ID to text
        generated_tokens.append(next_token)  # Add the token to the list of generated tokens

    return "".join(generated_tokens)  # Join all generated tokens into a single string


# Example of generating tokens using the function
tokens = generate(inputs, max_tokens=10)
print(tokens)

# Set padding token to be the same as EOS (End of Sequence) token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id  # Ensure pad token ID is set correctly

# Set padding strategy: Pad on the left so new tokens can be appended to the right
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# Multiple prompts of varying lengths to send to the model at once
prompts = [
    "The quick brown fox jumped over the",
    "The rain in Spain falls",
    "What comes up must",
]

# Tokenize the batch of prompts with padding enabled
inputs = tokenizer(prompts, padding=True, return_tensors="pt")

# Position IDs are used to tell the transformer the ordinal position of each token in the sequence
# For single input inference, this is just [0 .. n] but for batch inference, we need to handle padding tokens
attention_mask = inputs["attention_mask"]  # Extract the attention mask from the inputs
position_ids = attention_mask.long().cumsum(-1) - 1  # Create position IDs based on the attention mask
position_ids.masked_fill_(attention_mask == 0, 1)  # Set position IDs for padding tokens to 1 (or any other appropriate value)


# Function to generate a batch of tokens using past key values (batch processing)
def generate_batch_tokens_with_past(inputs):
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**inputs)  # Perform forward pass through the model
    
    logits = outputs.logits  # Get the logits (predicted scores for the next token)
    last_logits = logits[:, -1, :]  # Extract logits for the last token of each sequence in the batch
    next_token_ids = last_logits.argmax(dim=1)  # Get the token IDs with the highest scores (argmax)
    return next_token_ids, outputs.past_key_values  # Return the token IDs and past key values for caching


# Function to generate a batch of tokens with caching
def generate_batch(inputs, max_tokens):
    generate_tokens = [
        [] for _ in range(inputs["input_ids"].shape[0])  # Initialize an empty list for each batch item
    ]

    attention_mask = inputs["attention_mask"]  # Extract attention mask
    position_ids = attention_mask.long().cumsum(-1) - 1  # Generate position IDs based on attention mask
    position_ids.masked_fill_(attention_mask == 0, 1)  # Set position IDs for padding tokens to 1

    next_inputs = {
        "position_ids": position_ids,  # Include the position IDs
        **inputs  # Include the other inputs (input_ids, attention_mask, etc.)
    }

    # Generate tokens one by one up to the max_tokens
    for _ in range(max_tokens):
        next_token_ids, past_key_values = generate_token_with_past(next_inputs)  # Get next token IDs and past key values

        # Update inputs for the next generation step
        next_inputs = {
            "input_ids": next_token_ids.reshape((-1, 1)),  # Reshape the next token ID to match input format
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,  # Increment position IDs for the next token
            "attention_mask": torch.cat({
                next_inputs["attention_mask"],  # Extend attention mask by 1
                torch.ones(next_token_ids.shape[0], 1),  # Add a 1 for the new token
            }, dim=1),
            "past_key_values": past_key_values,  # Pass the cached key values for efficiency
        }

        next_tokens = tokenizer.batch_decode(next_token_ids)  # Decode the generated token IDs for each prompt in the batch
        for i, token in enumerate(next_tokens):
            generate_tokens[i].append(token)  # Add each token to the corresponding list in the batch

    # Return the generated text for each prompt in the batch
    return [" ".join(tokens) for tokens in generate_tokens]


# Example of generating tokens for a batch of prompts
batch_tokens = generate_batch(inputs, max_tokens=10)
print(batch_tokens)
