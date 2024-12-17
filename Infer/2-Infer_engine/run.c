/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

// Configuration structure for the Transformer model
typedef struct {
    int dim; // Transformer dimension (the size of the hidden states)
    int hidden_dim; // Dimension for feedforward layers (typically larger than dim)
    int n_layers; // Number of layers in the Transformer (stacked layers)
    int n_heads; // Number of attention heads in the multi-head attention mechanism
    int n_kv_heads; // Number of key/value heads (can be less than query heads for multi-query attention)
    int vocab_size; // Size of the vocabulary, usually 256 for byte-level tokenization
    int seq_len; // Maximum sequence length that the model can handle
} Config;

// Weights structure for the Transformer model
typedef struct {
    // Token embedding table for converting token IDs to dense vectors
    float* token_embedding_table;    // Shape: (vocab_size, dim)

    // Weights for RMS normalization layers
    float* rms_att_weight; // Weights for RMS normalization in attention layers (shape: (layer, dim))
    float* rms_ffn_weight; // Weights for RMS normalization in feedforward layers (shape: (layer, dim))

    // Weights for matrix multiplications in the attention mechanism
    // Note: dim == n_heads * head_size, where head_size is the size of each attention head
    float* wq; // Weights for query transformation (shape: (layer, dim, n_heads * head_size))
    float* wk; // Weights for key transformation (shape: (layer, dim, n_kv_heads * head_size))
    float* wv; // Weights for value transformation (shape: (layer, dim, n_kv_heads * head_size))
    float* wo; // Weights for output transformation after attention (shape: (layer, n_heads * head_size, dim))

    // Weights for the feedforward network (FFN) in each layer
    float* w1; // Weights for the first linear transformation in FFN (shape: (layer, hidden_dim, dim))
    float* w2; // Weights for the second linear transformation in FFN (shape: (layer, dim, hidden_dim))
    float* w3; // Weights for an additional linear transformation in FFN (shape: (layer, hidden_dim, dim))

    // Final RMS normalization weights applied after the last layer
    float* rms_final_weight; // Weights for final RMS normalization (shape: (dim,))

    // (Optional) Classifier weights for generating logits from the last layer's output
    float* wcls; // Weights for classification (shape: (dim, vocab_size))
} TransformerWeights;

// Run state structure to hold the current activations and intermediate values during the forward pass
typedef struct {
    // Current wave of activations
    float *x; // Activation at the current time step (shape: (dim,))
    float *xb; // Activation inside a residual branch (shape: (dim,))
    float *xb2; // Additional buffer for convenience (shape: (dim,))
    float *hb; // Buffer for hidden dimension in the feedforward network (shape: (hidden_dim,))
    float *hb2; // Additional buffer for hidden dimension in the feedforward network (shape: (hidden_dim,))
    float *q; // Query vector for attention (shape: (dim,))
    float *k; // Key vector for attention (shape: (dim,))
    float *v; // Value vector for attention (shape: (dim,))
    float *att; // Buffer for attention scores/values (shape: (n_heads, seq_len))
    float *logits; // Output logits for classification or generation

    // Key-Value cache for efficient attention computation
    float* key_cache;   // Cached keys for attention (shape: (layer, seq_len, dim))
    float* value_cache; // Cached values for attention (shape: (layer, seq_len, dim))
} RunState;

// Main structure for the Transformer model
typedef struct {
    Config config; // Hyperparameters of the architecture (the blueprint for the model)
    TransformerWeights weights; // Weights of the model, including embeddings and transformations
    RunState state; // Buffers for the "wave" of activations during the forward pass

    // Additional state needed for memory management
    int fd; // File descriptor for memory mapping the model weights
    float* data; // Pointer to the memory-mapped data for efficient access
    ssize_t file_size; // Size of the checkpoint file in bytes, used for memory management
} Transformer;

/**
 * @brief Reads a checkpoint file to restore the model's configuration and weights.
 *
 * This function opens a specified checkpoint file, reads the configuration header
 * into a provided Config structure, determines if the weights are shared based on
 * the vocabulary size, and calculates the size of the file. It then reopens the file
 * to memory map the weights into a data pointer for efficient access. Finally, it
 * calls a custom function to map the weights into the provided TransformerWeights
 * structure based on the configuration and shared weights status.
 *
 * @param checkpoint The path to the checkpoint file to be read.
 * @param config A pointer to a Config structure where the configuration will be stored.
 * @param weights A pointer to a TransformerWeights structure where the weights will be mapped.
 * @param fd A pointer to an integer that will hold the file descriptor for the opened file.
 * @param data A pointer to a float pointer that will point to the memory-mapped weights data.
 * @param file_size A pointer to a ssize_t variable that will hold the size of the checkpoint file.
 */
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    // Open the checkpoint file in binary read mode
    FILE *file = fopen(checkpoint, "rb");
    // Check if the file was opened successfully
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", checkpoint); 
        exit(EXIT_FAILURE); 
    }
    
    // Read the configuration header from the file into the config structure
    if (fread(config, sizeof(Config), 1, file) != 1) { 
        exit(EXIT_FAILURE); 
    }
    
    // Determine if the weights are shared based on the vocab_size
    // A negative vocab_size indicates unshared weights
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    // Store the absolute value of vocab_size in the config structure
    config->vocab_size = abs(config->vocab_size);
    
    // Move the file pointer to the end of the file to determine its size
    fseek(file, 0, SEEK_END); // Move file pointer to the end of the file
    // Get the current position of the file pointer, which is the file size in bytes
    *file_size = ftell(file); 
    // Close the file after obtaining the size
    fclose(file);
    
    // Open the checkpoint file again, this time to memory map the weights
    *fd = open(checkpoint, O_RDONLY); // Open in read-only mode
    // Check if the file descriptor was obtained successfully
    if (*fd == -1) { 
        fprintf(stderr, "open failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    
    // Memory map the file into the data pointer
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    // Check if the memory mapping was successful
    if (*data == MAP_FAILED) { 
        fprintf(stderr, "mmap failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    
    // Calculate the pointer to the weights data, which follows the config structure in memory
    float* weights_ptr = *data + sizeof(Config) / sizeof(float);
    
    // Call a function to map the weights into the provided weights structure
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

/**
 * @brief Allocates memory for the RunState structure based on the provided configuration.
 *
 * This function allocates memory for various components of the RunState structure
 * using calloc to ensure that the allocated memory is initialized to zero. This is
 * particularly useful for debugging with tools like Valgrind, which can report
 * uninitialized memory usage.
 *
 * @param s A pointer to the RunState structure that will hold the allocated memory.
 * @param p A pointer to the Config structure that contains configuration parameters
 *          used to determine the sizes of the allocated arrays.
 */
void malloc_run_state(RunState* s, Config* p) {
    // Calculate the dimension for key-value pairs based on the configuration
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    // Allocate memory for the input state vector (x) with size equal to the model dimension
    s->x = calloc(p->dim, sizeof(float)); // Input state vector

    // Allocate memory for the first bias vector (xb) with size equal to the model dimension
    s->xb = calloc(p->dim, sizeof(float)); // First bias vector

    // Allocate memory for the second bias vector (xb2) with size equal to the model dimension
    s->xb2 = calloc(p->dim, sizeof(float)); // Second bias vector

    // Allocate memory for the hidden state vector (hb) with size equal to the hidden dimension
    s->hb = calloc(p->hidden_dim, sizeof(float)); // Hidden state vector

    // Allocate memory for the second hidden state vector (hb2) with size equal to the hidden dimension
    s->hb2 = calloc(p->hidden_dim, sizeof(float)); // Second hidden state vector

    // Allocate memory for the query vector (q) with size equal to the model dimension
    s->q = calloc(p->dim, sizeof(float)); // Query vector

    // Allocate memory for the key cache with size equal to the number of layers, sequence length, and kv_dim
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)); // Key cache

    // Allocate memory for the value cache with size equal to the number of layers, sequence length, and kv_dim
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)); // Value cache

    // Allocate memory for the attention scores with size equal to the number of heads and sequence length
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float)); // Attention scores

    // Allocate memory for the logits with size equal to the vocabulary size
    s->logits = calloc(p->vocab_size, sizeof(float)); // Logits for vocabulary

    // Ensure all memory allocations were successful
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE); // Exit if any allocation fails
    }
}
void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

int main(int argc, char *argv[]){

    // Default parameters for the model and sampling
    // including weights and configs
    char *checkpoint_path = NULL;  // Path to the model checkpoint file (e.g., out/model.bin)
    // convert text to num
    char *tokenizer_path = "tokenizer.bin"; // Path to the tokenizer file
    // control the random
    float temperature = 1.0f;       // Sampling temperature: 0.0 = greedy (deterministic), 1.0 = original sampling
    // sample words that sum of possibility = topp
    float topp = 0.9f;              // Top-p value for nucleus sampling: 1.0 = off, 0.9 is a common choice but slower
    // length of generation
    int steps = 256;                // Number of steps to run for generation
    char *prompt = NULL;            // Prompt string for the model input
    unsigned long long rng_seed = 0; // Random number generator seed, default is 0 (time-based)
    char *mode = "generate";        // Mode of operation: "generate" or "chat"
    char *system_prompt = NULL;     // Optional system prompt for chat mode

    // Simple command-line argument parsing to override default parameters
    if (argc >= 2) { 
        checkpoint_path = argv[1]; // First argument is the checkpoint path
    } else { 
        error_usage(); // If no checkpoint path is provided, show usage error
    }

    // Loop through command-line arguments starting from the second argument
    for (int i = 2; i < argc; i += 2) {
        // Basic validation of command-line arguments
        if (i + 1 >= argc) { error_usage(); } // Ensure there is an argument after the flag
        if (argv[i][0] != '-') { error_usage(); } // Argument must start with a dash
        if (strlen(argv[i]) != 2) { error_usage(); } // Argument must be in the form -x (one dash, one letter)

        // Read in the arguments based on the flag
        if (argv[i][1] == 't') { 
            temperature = atof(argv[i + 1]); // Set temperature
        } else if (argv[i][1] == 'p') { 
            topp = atof(argv[i + 1]); // Set top-p value
        } else if (argv[i][1] == 's') { 
            rng_seed = atoi(argv[i + 1]); // Set random seed
        } else if (argv[i][1] == 'n') { 
            steps = atoi(argv[i + 1]); // Set number of steps
        } else if (argv[i][1] == 'i') { 
            prompt = argv[i + 1]; // Set prompt string
        } else if (argv[i][1] == 'z') { 
            tokenizer_path = argv[i + 1]; // Set tokenizer path
        } else if (argv[i][1] == 'm') { 
            mode = argv[i + 1]; // Set mode (generate or chat)
        } else if (argv[i][1] == 'y') { 
            system_prompt = argv[i + 1]; // Set system prompt for chat mode
        } else { 
            error_usage(); // If an unknown flag is encountered, show usage error
        }
    }

    // Parameter validation and overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL); // If seed is not set, use current time
    if (temperature < 0.0) temperature = 0.0; // Ensure temperature is not negative
    if (topp < 0.0 || 1.0 < topp) topp = 0.9; // Ensure top-p is within valid range
    if (steps < 0) steps = 0; // Ensure steps is not negative

    // Build the Transformer model using the checkpoint file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path); // Initialize the transformer with the model file
    if (steps == 0 || steps > transformer.config.seq_len) 
        steps = transformer.config.seq_len; // Override steps to maximum sequence length if necessary

    // Build the Tokenizer using the tokenizer file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size); // Initialize the tokenizer

    // Build the Sampler for generating samples
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed); // Initialize the sampler

    // Run the model based on the specified mode
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps); // Call generate function
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps); // Call chat function
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode); // Print error for unknown mode
        error_usage(); // Show usage error
    }

    // Cleanup memory and file handles
    free_sampler(&sampler); // Free resources allocated for the sampler
    free_tokenizer(&tokenizer); // Free resources allocated for the tokenizer
    free_transformer(&transformer); // Free resources allocated for the transformer
    return 0; // Return success
}