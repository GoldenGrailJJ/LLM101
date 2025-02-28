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

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

/**
 * @brief Maps the weights of the Transformer model from a memory pointer.
 *
 * This function assigns the appropriate memory locations for various weight matrices
 * in the Transformer model based on the provided configuration. It uses a pointer to
 * traverse the memory layout and assigns the weights to the corresponding fields in
 * the TransformerWeights structure.
 *
 * @param w A pointer to the TransformerWeights structure where the weights will be mapped.
 * @param p A pointer to the Config structure that contains configuration parameters
 *          used to determine the sizes of the weights.
 * @param ptr A pointer to the memory location where the weights are stored.
 * @param shared_weights An integer flag indicating whether the weights are shared (1) or not (0).
 */
/*
|-------------------------------|
| Token Embedding Table         |  (p->vocab_size * p->dim)
|-------------------------------|
| RMS Attention Weights         |  (n_layers * p->dim)
|-------------------------------|
| Query Weights (wq)            |  (n_layers * p->dim * (p->n_heads * head_size))
|-------------------------------|
| Key Weights (wk)              |  (n_layers * p->dim * (p->n_kv_heads * head_size))
|-------------------------------|
| Value Weights (wv)            |  (n_layers * p->dim * (p->n_kv_heads * head_size))
|-------------------------------|
| Output Weights (wo)           |  (n_layers * (p->n_heads * head_size) * p->dim)
|-------------------------------|
| RMS Feedforward Weights       |  (n_layers * p->dim)
|-------------------------------|
| First Feedforward Weights (w1)|  (n_layers * p->dim * p->hidden_dim)
|-------------------------------|
| Second Feedforward Weights(w2)| (n_layers * p->hidden_dim * p->dim)
|-------------------------------|
| Third Feedforward Weights (w3)|  (n_layers * p->dim * p->hidden_dim)
|-------------------------------|
| RMS Final Weights             |  (p->dim)
|-------------------------------|
| Skipped freq_cis_real         |  (p->seq_len * head_size / 2)
|-------------------------------|
| Skipped freq_cis_imag         |  (p->seq_len * head_size / 2)
|-------------------------------|
| CLS Weights (wcls)            |  (shared_weights ? w->token_embedding_table : ptr)
|-------------------------------|
*/
void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    // Calculate the size of each head based on the model dimension
    int head_size = p->dim / p->n_heads;

    // Use 64-bit integers to handle large parameter counts for models with 13B+ parameters
    unsigned long long n_layers = p->n_layers;

    // Map the token embedding table to the appropriate memory location
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim; // Move the pointer past the token embedding table

    // Map the RMS attention weights
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim; // Move the pointer past the RMS attention weights

    // Map the query weights
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size); // Move the pointer past the query weights

    // Map the key weights
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size); // Move the pointer past the key weights

    // Map the value weights
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size); // Move the pointer past the value weights

    // Map the output weights
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim; // Move the pointer past the output weights

    // Map the RMS feedforward network weights
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim; // Move the pointer past the RMS feedforward weights

    // Map the first feedforward layer weights
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim; // Move the pointer past the first feedforward layer weights

    // Map the second feedforward layer weights
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim; // Move the pointer past the second feedforward layer weights

    // Map the third feedforward layer weights
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim; // Move the pointer past the third feedforward layer weights

    // Map the RMS final weights
    w->rms_final_weight = ptr;
    ptr += p->dim; // Move the pointer past the RMS final weights

    // Skip the memory for frequency components used in RoPE (Rotary Positional Encoding)
    ptr += p->seq_len * head_size / 2; // Skip what used to be freq_cis_real
    ptr += p->seq_len * head_size / 2; // Skip what used to be freq_cis_imag

    // If weights are shared, point wcls to the token embedding table; otherwise, map to the next pointer
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

/**
 * @brief Frees the resources allocated for the Transformer structure.
 *
 * This function is responsible for cleaning up and releasing any memory or resources
 * that were allocated for the Transformer model. It ensures that all dynamically
 * allocated memory is properly freed to prevent memory leaks.
 *
 * @param t A pointer to the Transformer structure that needs to be freed.
 */
void free_transformer(Transformer* t) {
    // Close the memory mapping if it was successfully created
    if (t->data != MAP_FAILED) { 
        munmap(t->data, t->file_size); // Unmap the memory region associated with the Transformer data
    }
    
    // Close the file descriptor if it is valid
    if (t->fd != -1) { 
        close(t->fd); // Close the file descriptor to release the associated resources
    }
    
    // Free the RunState buffers associated with the Transformer
    free_run_state(&t->state); // Call a function to free the memory allocated for the RunState structure
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < size; i++) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (int i = 0; i < d; ++i) {
        float val = 0.0f;
        for (int j = 0; j < n; ++j) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos){
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    // 浮点数组，表示当前层的特征向量
    float *x = s->x;
    // 特征向量维度
    int dim = p->dim;
    // feedforward network的隐藏层维度
    int hidden_dim = p->hidden_dim;
    // n_kv_heads表示key和value的头数，n_heads表示query的头数
    int kv_dim = (p->dim * p->n_kv_heads) / n->heads;
    // 每个key/value头需要支持的查询头的数量
    // 比如 n_heads=12，n_kv_heads=4，那么每个key/value头需要支持3个查询头  
    int kv_mul = n_heads / n_kv_heads;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    // 数组：id to dense vector，根据id快速得到对应的dense vector
    // 数组：dense vector 变换为 feature vector
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    for(unsigned long long l = 0; l < p->n_layers; ++l){
        rmsnorm(s->xb, x, w->rms_attn_weight + l * dim, dim);

        int loff = p->seq_len * l * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; ++v) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int h;
        for (h = 0; h < p->n_heads; ++h) {
            // 找到第h个head的q，k，v
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            // 计算attention score
            for (int t = 0; t <= pos; ++t) {
                // 计算第t个位置的key，需要考虑层和头的影响
                // kv_mul表示key和value头的数量，kv_dim表示每个key和value头的维度
                // h / kv_mul表示当前head对应的key和value头的索引
                // 还是以 3 为例，h = 0, 1, 2 时对应的key和value头索引为 0
                float* k = s->key_cache + loff + t * kv_dim + h * kv_dim / kv_mul;
                float score = 0.0f;
                // q 和 k 的维度都是 head_size
                for (int i = 0; i < head_size; ++i) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                attn[t] = score;
            }
        }

        softmax(att, pos + 1);

        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));

        for (int t = 0; t <= pos; ++t) {
            float* v = s->value_cache + loff + t * kv_dim + h * head_size / kv_mul
            float score = att[t];
            for (int i = 0; i < head_size; ++i) {
                xb[i] += v[i] * score
            }
        }
    }
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

// TokenIndex struct stores a token's string and its corresponding ID.
typedef struct {
    char *str;  // The string representation of the token.
    int id;     // The unique identifier (ID) for the token.
} TokenIndex;

// Tokenizer struct holds the vocabulary and related data for tokenization.
// It also includes a sorted vocabulary for efficient lookups.
typedef struct {
    char** vocab;             // Array of strings representing the vocabulary.
    float* vocab_scores;      // Array of scores (probabilities, weights, etc.) for each vocabulary item.
    TokenIndex *sorted_vocab; // Sorted array of TokenIndex, used for efficient token lookups.
    int vocab_size;           // The total number of tokens in the vocabulary.
    unsigned int max_token_length; // The maximum length of a token (e.g., maximum byte length of token strings).
    unsigned char byte_pieces[512]; // Array to store all single-byte strings, typically for subword units or token pieces.
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

/**
 * @brief Builds a tokenizer by reading vocabulary and related data from a file.
 *
 * This function initializes a Tokenizer structure by reading the vocabulary size,
 * vocabulary strings, and their associated scores from a binary file. It allocates
 * memory for these components and initializes certain arrays. It also processes
 * single-byte strings for byte-pair encoding.
 *
 * @param t A pointer to the Tokenizer structure that will be populated.
 * @param tokenizer_path The path to the binary file containing the tokenizer data.
 * @param vocab_size The size of the vocabulary, indicating how many entries to read.
 *
 * @return void
 */
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // Set the vocabulary size to the given parameter value
    t->vocab_size = vocab_size; 

    // Allocate memory to store vocabulary strings and their corresponding scores
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));  // Allocate memory for vocabulary strings
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));  // Allocate memory for vocabulary scores

    // Initialize sorted_vocab lazily (not used for now)
    t->sorted_vocab = NULL;

    // Initialize byte_pieces array to represent all single-byte characters
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;   // Store the byte character at even indices
        t->byte_pieces[i * 2 + 1] = '\0';           // Null-terminate the byte string at odd indices
    }

    // Open the tokenizer file in binary read mode
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        // Handle file open failure and exit if the file cannot be loaded
        fprintf(stderr, "couldn't load %s\n", tokenizer_path); 
        exit(EXIT_FAILURE); // Exit on failure to open the file
    }

    // Read the maximum token length from the file
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        // Handle failed read operation and exit
        fprintf(stderr, "failed read\n"); 
        exit(EXIT_FAILURE);
    }

    int len;
    // Read the vocabulary entries (scores and strings) from the file
    for (int i = 0; i < vocab_size; i++) {
        // 1.Read the score for the current vocabulary item
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            // Handle failed read operation and exit
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }

        // 2.Read the length of the current vocabulary string
        if (fread(&len, sizeof(int), 1, file) != 1) {
            // Handle failed read operation and exit
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }

        // 3.Allocate memory for the current vocabulary string (including null terminator)
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            // Handle failed read operation and exit
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        
        // Null-terminate the string to ensure proper string handling
        t->vocab[i][len] = '\0';
    }

    // Close the file after reading all the necessary data
    fclose(file);
}

/**
 * @brief Frees the resources allocated for the Tokenizer structure.
 *
 * This function is responsible for cleaning up and releasing any dynamically allocated memory
 * for the Tokenizer, including the vocabulary strings, scores, and sorted vocabulary. 
 * It ensures that all dynamically allocated memory is properly freed to prevent memory leaks.
 *
 * @param t A pointer to the Tokenizer structure that needs to be freed.
 */
void free_tokenizer(Tokenizer* t) {
    // Loop through all vocabulary entries and free the memory allocated for each vocabulary string
    for (int i = 0; i < t->vocab_size; i++) { 
        free(t->vocab[i]);  // Free memory for each string in the vocabulary
    }
    
    // Free the memory allocated for the vocabulary array itself
    free(t->vocab);
    
    // Free the memory allocated for vocabulary scores
    free(t->vocab_scores);
    
    // Free the memory allocated for the sorted vocabulary structure
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        return (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

/*
 * Function: safe_printf
 * ---------------------
 * This function safely prints the input string `piece` to the console, but with certain restrictions.
 * Specifically, it ensures that only printable characters or whitespace are printed. 
 * This is useful in cases where `piece` may contain raw byte tokens, which might include control characters or other non-printable bytes.
 *
 * Parameters:
 *    piece   - The string to be printed.
 * 
 * Returns:
 *    None. It prints the string to the console if it's valid (i.e., contains printable characters).
 */
void safe_printf(char *piece) {
    // If the input piece is NULL, we do nothing and return.
    if (piece == NULL) { 
        return; 
    }

    // If the input piece is an empty string (first character is null terminator), do nothing and return.
    if (piece[0] == '\0') { 
        return; 
    }

    // If the string only contains one character, check if it's a printable character or whitespace.
    // This is to handle cases where piece may be a single raw byte or token.
    if (piece[1] == '\0') {
        // Convert the first byte to unsigned char to prevent sign extension issues.
        unsigned char byte_val = piece[0];
        
        // If the byte is neither a printable character nor whitespace, we don't print it.
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // The byte is not printable or whitespace, don't print it.
        }
    }

    // If the string passes the above checks, it's safe to print.
    printf("%s", piece);
}

/*
 * Function: str_lookup
 * ---------------------
 * This function searches for an exact match of the input string `str` in a sorted vocabulary array (`sorted_vocab`).
 * It uses binary search to find the string efficiently and returns the index of the matching token in the vocabulary.
 * If the string is not found, it returns -1.
 * 
 * Parameters:
 *    str           - The string to search for in the vocabulary.
 *    sorted_vocab  - The sorted array of `TokenIndex` structures representing the vocabulary.
 *    vocab_size    - The total size (number of elements) of the vocabulary.
 * 
 * Returns:
 *    The index (id) of the matching token in the vocabulary, or -1 if not found.
 */
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // Create a TokenIndex object with the search string as the key.
    // This object will be used as the search key for the binary search.
    TokenIndex tok = { .str = str };

    // Use binary search to find the token in the sorted vocabulary.
    // The `bsearch` function searches for the `tok` in the sorted vocabulary array.
    // It uses the `compare_tokens` function to compare elements.
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);

    // If the result is found (not NULL), return its ID (index) from the vocabulary.
    // If the result is NULL (not found), return -1.
    return res != NULL ? res->id : -1;
}

/*
 * Function: encode
 * ----------------
 * This function encodes an input text string into a sequence of tokens, using a vocabulary for encoding.
 * It optionally adds a Beginning-of-Sequence (BOS) token and an End-of-Sequence (EOS) token.
 * It also merges consecutive tokens into a new token if that merge has a higher score in the vocabulary.
 * 
 * Example:
 * 
 *    Tokenizer t;
 *    int tokens[100];
 *    int n_tokens = 0;
 *    encode(&t, "hello world", 1, 1, tokens, &n_tokens);
 * 
 * In the above example, the input text "hello world" will be tokenized, and the tokens will be stored
 * in the `tokens` array. The BOS and EOS tokens will be added because both `bos` and `eos` are set to 1.
 */
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // Check for null input text and exit with an error message if true
    if (text == NULL) { 
        fprintf(stderr, "cannot encode NULL text\n"); 
        exit(EXIT_FAILURE); 
    }

    // Lazy initialization of sorted vocabulary if it hasn't been done already
    if (t->sorted_vocab == NULL) {
        // Allocate memory for sorted vocabulary and copy contents
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        // Sort vocabulary based on tokens using a custom comparison function
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // Allocate a temporary buffer to store UTF-8 code points and merged token candidates
    // *2 for concatenation, +1 for the null terminator, +2 for possible UTF-8 encoding
    char* str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // Initialize token count to zero
    *n_tokens = 0;

    // Add optional BOS (=1) token if specified
    if (bos) tokens[(*n_tokens)++] = 1;

    // Optional dummy prefix handling (adds a space token to the input if not empty)
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Begin processing the input text as a sequence of UTF-8 bytes
    for (char *c = text; *c != '\0'; c++) {
        // Reset the buffer if the current byte is ASCII or the start of a new UTF-8 codepoint
        if ((*c & 0xC0) != 0x80) {
            // If not a continuation byte, reset the buffer to start a new codepoint
            str_len = 0;
        }

        // Append the current byte to the buffer
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0'; // Null-terminate the buffer

        // If the next byte is a continuation byte and we're not out of buffer space, keep appending
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // Now we've read a complete UTF-8 codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // If this codepoint exists in the vocabulary, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // Fallback to encoding each byte individually as a token
            // The "+3" accounts for the first three vocab elements (<unk>, <s>, </s>)
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        // Reset buffer length for the next codepoint
        str_len = 0;
    }

    // Merge the best consecutive pair of tokens based on their scores in the vocab
    while (1) {
        float best_score = -1e10; // Initialize best score to a very low value
        int best_id = -1;         // To store the id of the best merge token
        int best_idx = -1;        // To store the index of the best token pair

        // Check each consecutive pair of tokens
        for (int i = 0; i < (*n_tokens - 1); i++) {
            // Try to merge tokens[i] and tokens[i+1]
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            
            // If a valid pair is found and it has a higher score, record it
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        // If no valid pair was found, break the loop (no more merges possible)
        if (best_idx == -1) {
            break;
        }

        // Merge the best pair into a single token at the index best_idx
        tokens[best_idx] = best_id;

        // Shift the remaining tokens to remove the second token in the merged pair
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // Decrease the token count
    }

    // Add optional EOS (=2) token if specified
    if (eos) tokens[(*n_tokens)++] = 2;

    // Free the allocated memory for the string buffer
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; ++i) {
        if (probabilities[i] > max_p) {
            max_p = probabilities[i];
            max_i = i;
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

/**
 * @brief 比较两个 ProbIndex 对象的概率值，用于排序。
 *
 * 此函数适用于 `qsort` 等标准库排序函数，
 * 按照 `ProbIndex` 结构体中 `prob` 字段的大小进行降序排列。
 * 
 * @param a 指向第一个待比较的 ProbIndex 对象的指针。
 * @param b 指向第二个待比较的 ProbIndex 对象的指针。
 * @return 比较结果：
 *         - 返回 -1：如果 a 的 prob 大于 b 的 prob，表示 a 排在 b 前面（降序）。
 *         - 返回 1：如果 a 的 prob 小于 b 的 prob，表示 b 排在 a 前面。
 *         - 返回 0：如果 a 和 b 的 prob 相等。
 */
int compare(const void* a, const void* b) {
    // 将 void* 类型的指针强制转换为 ProbIndex* 类型
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;

    // 比较 prob 字段的值，按照降序排列
    if (a_->prob > b_->prob) return -1; // a_ 的 prob 更大，a_ 排在 b_ 前面
    if (a_->prob < b_->prob) return 1;  // b_ 的 prob 更大，b_ 排在 a_ 前面
    return 0;                           // prob 相等，位置保持不变
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // Top-p采样（或称“核采样”）从累积概率超过阈值 topp 的最小 token 集中进行采样。
    // 这种方法可以避免采样到概率非常低的 token，从而减少生成内容“跑偏”的风险。
    // 参数 coin 是一个随机数，范围为 [0, 1)，通常由 random_f32() 生成。

    int n0 = 0;
    // 按概率降序快速排序索引
    // 小于 (1 - topp) / (n - 1) 的值不能成为候选项
    // 为了提高效率，在排序之前将这些候选项剔除
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;       // 保存符合条件的 token 索引
            probindex[n0].prob = probabilities[i]; // 保存对应概率
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare); // 对候选项按概率降序排序

    // 截断列表，保留累积概率不超过 topp 的 token 集
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // 如果由于舍入误差无法满足条件，则默认考虑所有元素
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i; // 找到满足累积概率超过 topp 的位置
            break;        // 截断循环
        }
    }

    // 从截断后的 token 集中采样
    float r = coin * cumulative_prob; // 随机数映射到 [0, cumulative_prob) 范围
    float cdf = 0.0f;                 // 初始化累积分布函数
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {                // 判断随机数是否落入当前区间
            return probindex[i].index; // 返回对应的 token 索引
        }
    }
    return probindex[last_idx].index; // 如果出现舍入误差，返回最后一个 token
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;           // 设置词汇表大小
    sampler->temperature = temperature;        // 设置采样温度，用于控制生成的多样性
    sampler->topp = topp;                       // 设置 Top-p 采样阈值，用于控制概率累计范围
    sampler->rng_state = rng_seed;             // 初始化随机数生成器的种子状态
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));  // 分配存储概率的缓冲区
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);  // 释放分配给 probindex 的内存
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift 随机数生成器算法
    *state ^= *state >> 12;  // 位操作：右移 12 位并与当前状态异或
    *state ^= *state << 25;  // 位操作：左移 25 位并与当前状态异或
    *state ^= *state >> 27;  // 位操作：右移 27 位并与当前状态异或
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;  // 混淆并提取高 32 位
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;  // 生成随机浮点数
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

/**
 * @brief Generates text based on a prompt using the Transformer model and a Sampler.
 *
 * This function generates a sequence of tokens based on an initial text prompt. The tokens are 
 * iteratively predicted by the Transformer model, with the next token being either forced (if still 
 * processing the prompt) or sampled based on logits. The function continues generating tokens until 
 * the specified number of steps is reached or a special "beginning of sequence" (BOS) token is generated.
 *
 * @param transformer Pointer to the Transformer model used for generating tokens.
 * @param tokenizer Pointer to the Tokenizer used for encoding/decoding tokens.
 * @param sampler Pointer to the Sampler used for sampling the next token from logits.
 * @param prompt The initial prompt string to start the generation.
 * @param steps The number of tokens to generate.
 */
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    // 1.If prompt is NULL, use an empty string as default
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // 2.Encode the (string) prompt into a sequence of tokens
    int num_prompt_tokens = 0;
    //   Allocate memory for the prompt tokens array (+3 for '\0', BOS, and EOS tokens)
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); 
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        // Handle case where no prompt tokens are generated
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // 3.Start the main loop for generating tokens
    long start = 0;  // Used to time the execution, initialized after the first iteration
    int next;        // Will store the next token in the sequence
    int token = prompt_tokens[0]; // Initialize with the first token from the prompt
    int pos = 0;     // Position in the token sequence

    // Loop to generate tokens until reaching the specified number of steps
    while (pos < steps) {

        // Perform a forward pass through the transformer model to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // Advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // If still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // Otherwise, sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // Check for termination condition: BOS token (value 1) signifies end of sequence
        if (next == 1) { break; }

        // Decode the generated token back to string and print it
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);  // Print the token, skipping "unsafe" bytes
        fflush(stdout);      // Ensure the output is printed immediately
        token = next;        // Set the current token to the newly sampled token

        // Initialize the timer after the first iteration (which may be slower)
        if (start == 0) { start = time_in_ms(); }
    }

    // Print a newline after token generation
    printf("\n");

    // Report the achieved tokens per second (tok/s) based on the elapsed time
    if (pos > 1) {
        long end = time_in_ms();  // Get the end time
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    // Free the allocated memory for prompt tokens
    free(prompt_tokens);
}

/*
 * Function: read_stdin
 * ---------------------
 * This function reads a line of input from standard input (stdin) and stores it in a buffer. It ensures that the line does not exceed
 * a specified maximum buffer size and removes the newline character (`\n`) at the end of the input, if present.
 *
 * Parameters:
 *    guide    - A string (prompt) displayed to the user before reading input.
 *    buffer   - A pointer to the buffer where the input line will be stored.
 *    bufsize  - The size of the buffer. It ensures that no more than `bufsize - 1` characters are read.
 *
 * Returns:
 *    None. The function modifies the `buffer` to contain the input string (without the newline character).
 */
void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // Print the guide (prompt) message to the user, asking for input.
    printf("%s", guide);

    // Use fgets to read a line from stdin into the buffer, ensuring we don't exceed the buffer size.
    // fgets reads up to bufsize-1 characters, ensuring there's room for the null terminator.
    if (fgets(buffer, bufsize, stdin) != NULL) {
        // Get the length of the input string, including the newline character.
        size_t len = strlen(buffer);

        // Check if the last character is a newline and remove it.
        // fgets keeps the newline in the buffer if the input is shorter than the buffer size.
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // Replace the newline with null terminator
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.
void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {
        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
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
#endif