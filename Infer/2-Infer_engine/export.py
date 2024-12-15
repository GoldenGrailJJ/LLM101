"""
This script has functions and utilties for model export.
Basically, we have a bunch of versions of the model, and we
want to export them to .bin files to be read from and inferenced in C.

Among the "input" versions of PyTorch files/models:
- Official Llama 2 weights released by Meta
- Huggingface weights available on the hub
- llama2.c (this repo) trained models

Among the "output" versions of .bin files:
- v0: Legacy files of the original llama2.c repo (will eventually be DEPRECATED)
- v1-vN: Improved .bin files with a proper header, cache alignment, etc.

This script aspires to provide all of these conversions.
"""
import os
import gzip
import shutil
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer


# -----------------------------------------------------------------------------
# Common utilities

def serialize_fp32(file, tensor):
    """ 
    Writes a single float32 tensor to a binary file that is open in write-binary mode ('wb').

    Args:
        file: The file object where the tensor will be written.
        tensor: The PyTorch tensor to be serialized, expected to be in float32 format.
    """
    # Detach the tensor from the current computation graph, move it to CPU, flatten it to 1D,
    # convert it to float32, and then convert it to a NumPy array.
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    
    # Pack the NumPy array into a binary format using struct, where 'f' indicates float32.
    b = struct.pack(f'{len(d)}f', *d)
    
    # Write the packed binary data to the specified file.
    file.write(b)

def serialize_int8(file, tensor):
    """ 
    Writes a single int8 tensor to a binary file that is open in write-binary mode ('wb').

    Args:
        file: The file object where the tensor will be written.
        tensor: The PyTorch tensor to be serialized, expected to be in int8 format.
    """
    # Detach the tensor from the current computation graph, move it to CPU, flatten it to 1D,
    # convert it to int8 format using NumPy.
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    
    # Pack the NumPy array into a binary format using struct, where 'b' indicates int8.
    b = struct.pack(f'{len(d)}b', *d)
    
    # Write the packed binary data to the specified file.
    file.write(b)

def quantize_q80(w, group_size):
    """
    Takes a tensor and returns its Q8_0 quantized version, which is a symmetric quantization
    into int8 format, with values in the range [-127, 127].

    Args:
        w: The input tensor to be quantized.
        group_size: The size of the groups for quantization.

    Returns:
        int8val: The quantized tensor in int8 format.
        scale: The scaling factors used for quantization.
        maxerr: The maximum quantization error across all groups.
    """
    # Ensure that the number of elements in the tensor is a multiple of the group size.
    assert w.numel() % group_size == 0
    
    # Store the original shape of the tensor for later use.
    ori_shape = w.shape
    
    # Convert the tensor to float32 for quantization calculations.
    w = w.float() 
    
    # Reshape the tensor into groups of the specified size.
    w = w.reshape(-1, group_size)
    
    # Find the maximum absolute value in each group.
    wmax = torch.abs(w).max(dim=1).values
    
    # Calculate the scaling factor such that the float value equals quantized value times scale.
    scale = wmax / 127.0
    
    # Scale the tensor into the range [-127, 127].
    quant = w / scale[:, None]
    
    # Round the scaled values to the nearest integer and convert to int8.
    int8val = torch.round(quant).to(torch.int8)
    
    # Dequantize by rescaling the int8 values back to float32.
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    
    # Reshape the dequantized values back to the original group shape.
    fp32valr = fp32val.reshape(-1, group_size)
    
    # Calculate the maximum error in each group by comparing the dequantized values to the original tensor.
    err = torch.abs(fp32valr - w).max(dim=1).values
    
    # Find the maximum error across all groups.
    maxerr = err.max().item()
    
    # Return the quantized tensor, scaling factors, and maximum error.
    return int8val, scale, maxerr


# -----------------------------------------------------------------------------
# Output

def legacy_export(model, filepath):
    """
    Export the model weights to a legacy binary format (version v0) used by llama2.c.

    Args:
        model: The model instance containing the weights to be exported.
        filepath (str): The path where the binary file will be saved.
    """
    
    # Open the output file in binary write mode.
    out_file = open(filepath, 'wb')

    # First, write out the header information for the binary file.
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]  # Get the hidden dimension from the first layer's feed-forward weights.
    p = model.params  # Access the model parameters.
    
    # Check if the token embeddings and output weights are shared.
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    
    # In the legacy format, use negative vocabulary size as a flag for shared classifier.
    if not shared_classifier:
        p.vocab_size = -p.vocab_size  # Negate the vocabulary size if not shared.

    # Determine the number of key-value heads, defaulting to the number of heads if not specified.
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    
    # Pack the header information into a binary format.
    # i*7 means packing 7 int to binary string
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    
    # Write the header to the output file.
    out_file.write(header)

    # Next, write out the embedding weights of the model.
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # Now, write out the weights for all layers in the model.
    
    # Write attention weights for each layer.
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)  # Write attention normalization weights.
    
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)  # Write query weights.
    
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)  # Write key weights.
    
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)  # Write value weights.
    
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)  # Write output weights.

    # Write feed-forward network weights for each layer.
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)  # Write feed-forward normalization weights.
    
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)  # Write first feed-forward layer weights.
    
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)  # Write second feed-forward layer weights.
    
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)  # Write third feed-forward layer weights.

    # Write the final normalization weights.
    serialize_fp32(out_file, model.norm.weight)

    # Write the frequency components for cosine and sine.
    serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])  # Write cosine frequencies.
    serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])  # Write sine frequencies.

    # Write the final classifier weights if the classifier is not shared.
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    # Close the output file after writing all weights.
    out_file.close()
    
    # Print a message indicating that the file has been written successfully.
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# Load / import functions

def load_checkpoint(checkpoint):
    """
    Load the model from the provided checkpoint file.

    Args:
        checkpoint (str): The path to the checkpoint file (.pt) to load.

    Returns:
        model: The loaded model instance with the state dictionary applied.
    """

    # Load the provided model checkpoint from the specified file path.
    # The model is loaded onto the CPU to ensure compatibility regardless of the original device.
    checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    
    # Extract model configuration parameters from the checkpoint.
    # 'model_args' contains the necessary arguments to initialize the model.
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    
    # Create a new instance of the Transformer model using the extracted configuration.
    model = Transformer(gptconf)
    
    # Retrieve the state dictionary containing the model's parameters from the checkpoint.
    state_dict = checkpoint_dict['model']
    
    # Define a prefix that may be present in the state dictionary keys that we want to remove.
    unwanted_prefix = '_orig_mod.'
    
    # Iterate over the items in the state dictionary to check for unwanted prefixes.
    for k, v in list(state_dict.items()):
        # If the key starts with the unwanted prefix, remove the prefix from the key.
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Load the state dictionary into the model.
    # 'strict=False' allows for loading parameters even if some keys do not match exactly.
    model.load_state_dict(state_dict, strict=False)
    
    # Set the model to evaluation mode, which disables dropout and batch normalization layers.
    model.eval()
    
    # Return the loaded model instance.
    return model

def load_meta_model(model_path):
    """
    Load a model from the specified Meta LLaMA model path.

    Args:
        model_path (str): The path to the directory containing the model files.

    Returns:
        model: The loaded Transformer model instance.
    """

    # Construct the path to the parameters JSON file.
    params_path = os.path.join(model_path, 'params.json')
    
    # Open and read the parameters JSON file to extract model configuration.
    with open(params_path) as f:
        params = json.load(f)  # Load the JSON data into a Python dictionary.
        print(params)  # Print the parameters for debugging purposes.

    # Find all model weight files that match the pattern 'consolidated.*.pth' in the model path.
    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    
    # Load each model weight file into a list of models, mapping them to the CPU.
    models = [torch.load(p, map_location='cpu') for p in model_paths]

    def concat_weights(models):
        """
        Concatenate weights from multiple models into a single state dictionary.

        Args:
            models (list): A list of model state dictionaries.

        Returns:
            state_dict (dict): A dictionary containing concatenated weights.
        """
        state_dict = {}  # Initialize an empty dictionary to hold the concatenated weights.
        
        # Iterate over the keys in the first model's state dictionary.
        for name in list(models[0]):
            # Extract the tensors corresponding to the current key from all models.
            tensors = [model[name] for model in models]
            
            # If there is only one tensor or the tensor is 1-dimensional, directly assign it.
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            
            # Determine the axis along which to concatenate based on the key name.
            is_axis_1 = (
                name.startswith('tok_embeddings.')  # Check if the key is related to token embeddings.
                or name.endswith('.attention.wo.weight')  # Check for attention output weights.
                or name.endswith('.feed_forward.w2.weight')  # Check for feed-forward layer weights.
            )
            axis = 1 if is_axis_1 else 0  # Set the axis for concatenation.
            
            # Concatenate the tensors along the specified axis and store in the state dictionary.
            state_dict[name] = torch.cat(tensors, dim=axis)
            
            # Delete the key from each model to free up memory.
            for model in models:
                del model[name]
        
        return state_dict  # Return the concatenated state dictionary.

    # Call the concat_weights function to combine weights from all models.
    state_dict = concat_weights(models)
    
    # Delete the models list to free up memory.
    del models

    # Initialize a new ModelArgs object to hold model configuration.
    config = ModelArgs()
    
    # Set model configuration parameters from the loaded JSON parameters.
    config.dim = params["dim"]  # Set the dimensionality of the model.
    config.n_layers = params["n_layers"]  # Set the number of layers in the model.
    config.n_heads = params["n_heads"]  # Set the number of attention heads.
    config.n_kv_heads = params.get('n_kv_heads') or params['n_heads']  # Set key-value heads.
    config.multiple_of = params["multiple_of"]  # Set the multiple of dimensions.
    config.norm_eps = params["norm_eps"]  # Set the normalization epsilon value.

    # Set additional configuration parameters based on the state dictionary.
    config.vocab_size = state_dict['tok_embeddings.weight'].shape[0]  # Set vocabulary size.
    config.max_seq_len = 2048  # Set maximum sequence length.

    # Create a new Transformer model instance using the configuration.
    model = Transformer(config)

    # Assign the token embeddings and normalization weights from the state dictionary to the model.
    model.tok_embeddings.weight = nn.Parameter(state_dict['tok_embeddings.weight'])
    model.norm.weight = nn.Parameter(state_dict['norm.weight'])

    # Iterate over each layer in the model to assign the corresponding weights.
    for layer in model.layers:
        i = layer.layer_id  # Get the layer ID for indexing.
        layer.attention_norm.weight = nn.Parameter(state_dict[f'layers.{i}.attention_norm.weight'])
        layer.attention.wq.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wq.weight'])
        layer.attention.wk.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wk.weight'])
        layer.attention.wv.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wv.weight'])
        layer.attention.wo.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wo.weight'])
        layer.ffn_norm.weight = nn.Parameter(state_dict[f'layers.{i}.ffn_norm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w1.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w2.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w3.weight'])

    # Assign the output layer weights from the state dictionary to the model.
    model.output.weight = nn.Parameter(state_dict['output.weight'])
    
    # Set the model to evaluation mode, which disables dropout and batch normalization.
    model.eval()
    
    # Return the fully constructed and loaded model instance.
    return model

def load_hf_model(model_path):
    """
    Load a Hugging Face model from the specified path and convert it to a custom Transformer model.

    Args:
        model_path (str): The path to the directory containing the Hugging Face model files.

    Returns:
        model: The loaded Transformer model instance.
    """

    # Attempt to import the AutoModelForCausalLM class from the transformers library.
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        # If the transformers package is not installed, print an error message and return None.
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # Load the Hugging Face model using the specified model path.
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Retrieve the state dictionary (weights) of the loaded Hugging Face model.
    hf_dict = hf_model.state_dict()

    # Convert the Hugging Face model configuration to a custom ModelArgs object.
    config = ModelArgs()
    config.dim = hf_model.config.hidden_size  # Set the hidden size of the model.
    config.n_layers = hf_model.config.num_hidden_layers  # Set the number of layers in the model.
    config.n_heads = hf_model.config.num_attention_heads  # Set the number of attention heads.
    config.n_kv_heads = hf_model.config.num_attention_heads  # Set the number of key-value heads.
    config.vocab_size = hf_model.config.vocab_size  # Set the vocabulary size.
    config.hidden_dim = hf_model.config.intermediate_size  # Set the hidden dimension for feed-forward layers.
    config.norm_eps = hf_model.config.rms_norm_eps  # Set the epsilon value for normalization.
    config.max_seq_len = hf_model.config.max_position_embeddings  # Set the maximum sequence length.

    # Create a new Transformer model instance using the configuration.
    model = Transformer(config)

    # Set the token embeddings and normalization weights from the Hugging Face model to the custom model.
    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    # Define a function to reverse the permutation of weights for query and key projections.
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        # Reshape and transpose the weight tensor to reverse the permutation applied by Hugging Face.
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Iterate over each layer in the custom Transformer model to set the corresponding weights.
    for layer in model.layers:
        i = layer.layer_id  # Get the layer ID for indexing.
        
        # Set the weights for attention normalization and attention mechanisms.
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        layer.attention.wq.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight']))
        layer.attention.wk.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']))
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        
        # Set the weights for feed-forward network normalization and layers.
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])

    # Set the output layer weights from the Hugging Face model to the custom model.
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
    
    # Set the model to evaluation mode, which disables dropout and batch normalization.
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())

    # Return the fully constructed and loaded Transformer model instance.
    return model

# -----------------------------------------------------------------------------
# CLI entrypoint
if __name__ == "__main__":  # Check if the script is being run directly (not imported as a module)

    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    
    # Add a positional argument for the output file path
    parser.add_argument("filepath", type=str, help="the output filepath")
    
    # Add an optional argument for the version of the export (default is 0)
    parser.add_argument("--version", default=0, type=int, help="the version to export with")
    
    # Add an optional argument for the data type of the model (default is 'fp32')
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    
    # Create a mutually exclusive group for model input options (only one can be provided)
    group = parser.add_mutually_exclusive_group(required=True)
    
    # Add an argument for loading a model checkpoint from a .pt file
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    
    # Add an argument for loading a Meta LLaMA model from a specified path
    group.add_argument("--meta-llama", type=str, help="meta llama model path")
    
    # Add an argument for loading a Hugging Face model from a specified path
    group.add_argument("--hf", type=str, help="huggingface model path")
    
    # Parse the command-line arguments and store them in 'args'
    args = parser.parse_args()
    
    # Map the dtype argument to the corresponding PyTorch data type
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Load the model based on the provided input argument
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)  # Load model from checkpoint
    elif args.meta_llama:
        model = load_meta_model(args.meta_llama)  # Load model from Meta LLaMA path
    elif args.hf:
        model = load_hf_model(args.hf)  # Load model from Hugging Face path

    # Check if the model was successfully loaded; if not, raise an error
    if model is None:
        parser.error("Can't load input model!")  # Display error message if model loading fails

    # Export the loaded model to the specified file path with the given version and data type
    model_export(model, args.filepath, args.version, args.dtype)