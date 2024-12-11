import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors = "pt")

def generate_token_with_past(inputs):
    with torch.no_grad()：
        outputs = model(**inputs)
    
    logits = outputs.logits
    last_logits = logits[0, -1, : ]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values


generated_tokens = []
next_inputs = inputs

for _ in range(10):
    next_token_id, past_key_values = \
        generate_token_with_past(next_inputs)
    
    next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor(([1]))],
            dim=1),
        "past_key_values" : past_key_values,
    }

    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)