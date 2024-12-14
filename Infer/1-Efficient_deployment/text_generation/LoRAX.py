import asyncio
import json
import time
from typing import List

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pydantic import BaseModel, constr

from lorax import AsyncClient, Client
from utils import endpoint_url, headers

# Initialize synchronous client for API interaction
client = Client(endpoint_url, headers=headers)

# Part 1: KV Cache Demonstration

# Measure the time taken to generate text synchronously using client.generate
t0 = time.time()
resp = client.generate("What is deep learning?", max_new_tokens=32)
duration_s = time.time() - t0

print(resp.generated_text)
print("\n\n----------")
print("Request duration (s):", duration_s)

# Streaming generation: Each token is streamed as it is generated
durations_s = []

t0 = time.time()
for resp in client.generate_stream("What is deep learning?", max_new_tokens=32):
    durations_s.append(time.time() - t0)
    # Print the token text if it is not special (e.g., punctuation, end token)
    if not resp.token.special:
        print(resp.token.text, sep="", end="", flush=True)
    t0 = time.time()

print("\n\n\n----------")
print("Time to first token (TTFT) (s):", durations_s[0])
print("Throughput (tok / s):", (len(durations_s) - 1) / sum(durations_s[1:]))

# Part 2: Continuous Batching for Multiple Prompts

# Color codes for text output formatting in terminal
color_codes = [
    "31",  # red
    "32",  # green
    "34",  # blue
]

# Helper function to format text output with color
def format_text(text, i):
    return f"\x1b[{color_codes[i]}m{text}\x1b[0m"

# Initialize asynchronous client for API interaction
async_client = AsyncClient(endpoint_url, headers=headers)

# List to store durations for each prompt in multiple streams
durations_s = [[], [], []]

# Asynchronous function to handle generation streaming for each prompt
async def run(max_new_tokens, i):
    t0 = time.time()
    async for resp in async_client.generate_stream("What is deep learning?", max_new_tokens=max_new_tokens):
        durations_s[i].append(time.time() - t0)
        print(format_text(resp.token.text, i), sep="", end="", flush=True)
        t0 = time.time()

# Start asynchronous batch execution for three different max_new_tokens values
t0 = time.time()
all_max_new_tokens = [100, 10, 10]
await asyncio.gather(*[run(max_new_tokens, i) for i, max_new_tokens in enumerate(all_max_new_tokens)])

# Output performance statistics
print("\n\n\n----------")
print("Time to first token (TTFT) (s):", [s[0] for s in durations_s])
print("Throughput (tok / s):", [(len(s) - 1) / sum(s[1:]) for s in durations_s])
print("Total duration (s):", time.time() - t0)

# Part 3: Multi-LoRA (Using Multiple Adapters for Generation)

# Function to generate text using a specific adapter (LoRA-based fine-tuning)
def run_with_adapter(prompt, adapter_id):
    durations_s = []

    t0 = time.time()
    for resp in client.generate_stream(
        prompt, 
        adapter_id=adapter_id,
        adapter_source="hub",  # Adapter source from the hub
        max_new_tokens=64,  # Number of tokens to generate
    ):
        durations_s.append(time.time() - t0)
        if not resp.token.special:
            print(resp.token.text, sep="", end="", flush=True)
        t0 = time.time()

    # Output performance statistics for adapter-based generation
    print("\n\n\n----------")
    print("Time to first token (TTFT) (s):", durations_s[0])
    print("Throughput (tok / s):", (len(durations_s) - 1) / sum(durations_s[1:]))

# Example prompts for Multi-LoRA generation tasks
pt_hellaswag_processed = """You are provided with an incomplete passage below. Please read the passage and then finish it with an appropriate response. For example:

### Passage: My friend and I think alike. We

### Ending: often finish each other's sentences.

Now please finish the following passage:

### Passage: {ctx}

### Ending: """
ctx = "Numerous people are watching others on a field. Trainers are playing frisbee with their dogs. the dogs"

# Run the Multi-LoRA task for Hellaswag adapter
run_with_adapter(pt_hellaswag_processed.format(ctx=ctx), adapter_id="predibase/hellaswag_processed")

# Example for another Multi-LoRA task: CNN summarization task
pt_cnn = """You are given a news article below. Please summarize the article, including only its highlights.

### Article: {article}

### Summary: """
article = "(CNN)Former Vice President Walter Mondale was released from the Mayo Clinic on Saturday after being admitted with influenza, hospital spokeswoman Kelley Luckstein said. \"He's doing well. We treated him for flu and cold symptoms and he was released today,\" she said..."

# Run the Multi-LoRA task for CNN summarization adapter
run_with_adapter(pt_cnn.format(article=article), adapter_id="predibase/cnn")

# Example for Named Entity Recognition (NER) task using another adapter
pt_conllpp = """
Your task is a Named Entity Recognition (NER) task. Predict the category of
each entity, then place the entity into the list associated with the 
category in an output JSON payload. Below is an example:

Input: EU rejects German call to boycott British lamb . Output: {{"person":
[], "organization": ["EU"], "location": [], "miscellaneous": ["German",
"British"]}}

Now, complete the task.

Input: {inpt} Output:"""
inpt = "Only France and Britain backed Fischler 's proposal ."

# Run the Multi-LoRA task for NER adapter
run_with_adapter(pt_conllpp.format(inpt=inpt), adapter_id="predibase/conllpp")

# Part 4: Structured Generation Using Pydantic Models

# Define a Pydantic model for structured data generation (Person object)
class Person(BaseModel):
    name: constr(max_length=10)
    age: int

# Get the schema of the Person model
schema = Person.schema()
schema

# Generate a person description in structured format
resp = client.generate(
    "Create a person description for me", 
    response_format={"type": "json_object", "schema": schema}
)
print(json.loads(resp.generated_text))

# Another structured generation example: Named Entity Recognition task
prompt_template = """
Your task is a Named Entity Recognition (NER) task. Predict the category of
each entity, then place the entity into the list associated with the 
category in an output JSON payload. Below is an example:

Input: EU rejects German call to boycott British lamb . Output: {{"person":
[], "organization": ["EU"], "location": [], "miscellaneous": ["German",
"British"]}}

Now, complete the task.

Input: {input} Output:"""

# Generate structured NER task output using a model adapter
resp = client.generate(
    prompt_template.format(input="Only France and Britain backed Fischler 's proposal ."),  
    max_new_tokens=128,
)
print(resp.generated_text)

# Define the output schema for the NER result using Pydantic
class Output(BaseModel):
    person: List[str]
    organization: List[str]
    location: List[str]
    miscellaneous: List[str]

# Get the schema of the Output model
schema = Output.schema()
schema

# Generate the NER result in structured format using the model schema
resp = client.generate(
    prompt_template.format(input="Only France and Britain backed Fischler 's proposal ."),
    response_format={
        "type": "json_object",
        "schema": schema,
    },
    max_new_tokens=128,
)
print(json.loads(resp.generated_text))

# Another example with a specific adapter for NER tasks
resp = client.generate(
    prompt_template.format(input="Only France and Britain backed Fischler 's proposal ."),
    adapter_id="predibase/conllpp",
    adapter_source="hub",
    max_new_tokens=128,
)
print(json.loads(resp.generated_text))
