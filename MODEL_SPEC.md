# Concept Predictor GGUF Specification v1.0

> **Pioneering Format**: This specification defines a new GGUF architecture type `concept_predictor` for embedding-space prediction models, enabling edge deployment of JEPA-style concept predictors alongside quantized LLMs.

## Overview

The Concept Predictor is a lightweight MLP that maps text embeddings to code concept embeddings. It works in tandem with:
1. A **text encoder** (e.g., GTE-Large) to encode the user's query
2. A **concept bank** to retrieve similar code examples
3. A **code LLM** (e.g., Qwen3-Coder) to generate the final code

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query: "implement binary search"                          │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐                                          │
│  │  Text Encoder    │  ← External (GTE-Large, 1024-dim)        │
│  │  (not in GGUF)   │                                          │
│  └────────┬─────────┘                                          │
│           │ [1024-dim vector]                                   │
│           ▼                                                     │
│  ┌──────────────────┐                                          │
│  │ CONCEPT PREDICTOR│  ← THIS GGUF FILE                        │
│  │    (MLP)         │     ~4MB, runs on CPU                    │
│  └────────┬─────────┘                                          │
│           │ [256-dim normalized vector]                         │
│           ▼                                                     │
│  ┌──────────────────┐                                          │
│  │  Concept Bank    │  ← Separate file (concept_bank.pt)       │
│  │  (Vector Search) │     cosine similarity lookup             │
│  └────────┬─────────┘                                          │
│           │ [top-k code examples]                               │
│           ▼                                                     │
│  ┌──────────────────┐                                          │
│  │   Code LLM       │  ← Separate GGUF (e.g., Qwen2.5-Coder)   │
│  │  + Examples      │                                          │
│  └────────┬─────────┘                                          │
│           │                                                     │
│           ▼                                                     │
│  Generated Code                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## GGUF Metadata Keys

### Required General Metadata

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `general.architecture` | string | Must be `"concept_predictor"` | `"concept_predictor"` |
| `general.name` | string | Model name | `"concept-first-codegen"` |
| `general.author` | string | Author/organization | `"rileyseaburg"` |
| `general.license` | string | License identifier | `"MIT"` |

### Required Architecture Metadata

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `concept_predictor.text_encoder` | string | HuggingFace model ID for text encoder | `"Alibaba-NLP/gte-large-en-v1.5"` |
| `concept_predictor.text_dim` | uint64 | Input dimension (text embedding size) | `1024` |
| `concept_predictor.concept_dim` | uint64 | Output dimension (concept embedding size) | `256` |
| `concept_predictor.hidden_dim` | uint64 | Hidden layer dimension | `1024` |
| `concept_predictor.num_layers` | uint64 | Number of linear layers | `3` |
| `concept_predictor.activation` | string | Activation function | `"gelu"` |
| `concept_predictor.quantization` | string | Weight quantization type | `"f32"`, `"f16"`, `"q8_0"` |

### Optional Metadata

| Key | Type | Description |
|-----|------|-------------|
| `general.description` | string | Model description |
| `general.url` | string | Model homepage URL |
| `concept_predictor.dropout` | float32 | Dropout rate used during training |
| `concept_predictor.norm_type` | string | Normalization type (`"layernorm"`, `"rmsnorm"`) |

## Tensor Layout

The projector MLP consists of 3 linear layers with LayerNorm:

```
Layer 0: Linear(text_dim → hidden_dim)
         LayerNorm(hidden_dim)
         GELU + Dropout

Layer 1: Linear(hidden_dim → hidden_dim)  
         LayerNorm(hidden_dim)
         GELU + Dropout

Layer 2: Linear(hidden_dim → concept_dim)
         [output is L2-normalized]
```

### Tensor Names

| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `projector.0.weight` | `[hidden_dim, text_dim]` | First linear layer weights |
| `projector.0.bias` | `[hidden_dim]` | First linear layer bias |
| `projector.2.weight` | `[hidden_dim]` | First LayerNorm weight (gamma) |
| `projector.2.bias` | `[hidden_dim]` | First LayerNorm bias (beta) |
| `projector.4.weight` | `[hidden_dim, hidden_dim]` | Second linear layer weights |
| `projector.4.bias` | `[hidden_dim]` | Second linear layer bias |
| `projector.6.weight` | `[hidden_dim]` | Second LayerNorm weight |
| `projector.6.bias` | `[hidden_dim]` | Second LayerNorm bias |
| `projector.8.weight` | `[concept_dim, hidden_dim]` | Final linear layer weights |
| `projector.8.bias` | `[concept_dim]` | Final linear layer bias |

**Note**: Layer indices (0, 2, 4, 6, 8) account for nn.Sequential with activation/dropout layers.

## Inference Pseudocode

```python
def concept_predictor_forward(text_embedding, weights):
    """
    Forward pass for concept predictor.
    
    Args:
        text_embedding: [text_dim] float32 vector from text encoder
        weights: dict of tensors loaded from GGUF
    
    Returns:
        [concept_dim] L2-normalized concept embedding
    """
    x = text_embedding  # [text_dim]
    
    # Layer 0: Linear + LayerNorm + GELU
    x = x @ weights["projector.0.weight"].T + weights["projector.0.bias"]
    x = layer_norm(x, weights["projector.2.weight"], weights["projector.2.bias"])
    x = gelu(x)
    
    # Layer 1: Linear + LayerNorm + GELU  
    x = x @ weights["projector.4.weight"].T + weights["projector.4.bias"]
    x = layer_norm(x, weights["projector.6.weight"], weights["projector.6.bias"])
    x = gelu(x)
    
    # Layer 2: Linear (final projection)
    x = x @ weights["projector.8.weight"].T + weights["projector.8.bias"]
    
    # L2 normalize output
    x = x / norm(x)
    
    return x


def gelu(x):
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))


def layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization."""
    mean = mean(x)
    var = var(x)
    x_norm = (x - mean) / sqrt(var + eps)
    return weight * x_norm + bias
```

## C/C++ Reference Implementation

```c
// concept_predictor.h
#ifndef CONCEPT_PREDICTOR_H
#define CONCEPT_PREDICTOR_H

#include <stdint.h>

typedef struct {
    int text_dim;
    int hidden_dim;
    int concept_dim;
    
    // Layer 0
    float* linear0_weight;  // [hidden_dim, text_dim]
    float* linear0_bias;    // [hidden_dim]
    float* norm0_weight;    // [hidden_dim]
    float* norm0_bias;      // [hidden_dim]
    
    // Layer 1
    float* linear1_weight;  // [hidden_dim, hidden_dim]
    float* linear1_bias;    // [hidden_dim]
    float* norm1_weight;    // [hidden_dim]
    float* norm1_bias;      // [hidden_dim]
    
    // Layer 2 (output)
    float* linear2_weight;  // [concept_dim, hidden_dim]
    float* linear2_bias;    // [concept_dim]
} concept_predictor_t;

// Load from GGUF file
concept_predictor_t* concept_predictor_load(const char* path);

// Run inference
// Input: text_embedding[text_dim] from external text encoder
// Output: concept_embedding[concept_dim] (L2 normalized)
void concept_predictor_forward(
    concept_predictor_t* model,
    const float* text_embedding,
    float* concept_embedding
);

// Free model
void concept_predictor_free(concept_predictor_t* model);

#endif
```

```c
// concept_predictor.c
#include "concept_predictor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void linear(float* out, const float* in, 
                   const float* weight, const float* bias,
                   int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_dim; j++) {
            sum += weight[i * in_dim + j] * in[j];
        }
        out[i] = sum;
    }
}

static void layer_norm(float* x, const float* weight, const float* bias, int dim) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    
    float var = 0.0f;
    for (int i = 0; i < dim; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= dim;
    
    float std = sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++) {
        x[i] = weight[i] * (x[i] - mean) / std + bias[i];
    }
}

static void gelu(float* x, int dim) {
    for (int i = 0; i < dim; i++) {
        float val = x[i];
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        x[i] = val * cdf;
    }
}

static void l2_normalize(float* x, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += x[i] * x[i];
    norm = sqrtf(norm + 1e-12f);
    for (int i = 0; i < dim; i++) x[i] /= norm;
}

void concept_predictor_forward(
    concept_predictor_t* model,
    const float* text_embedding,
    float* concept_embedding
) {
    float* hidden = (float*)malloc(model->hidden_dim * sizeof(float));
    
    // Layer 0: Linear -> LayerNorm -> GELU
    linear(hidden, text_embedding, 
           model->linear0_weight, model->linear0_bias,
           model->hidden_dim, model->text_dim);
    layer_norm(hidden, model->norm0_weight, model->norm0_bias, model->hidden_dim);
    gelu(hidden, model->hidden_dim);
    
    // Layer 1: Linear -> LayerNorm -> GELU
    float* hidden2 = (float*)malloc(model->hidden_dim * sizeof(float));
    linear(hidden2, hidden,
           model->linear1_weight, model->linear1_bias,
           model->hidden_dim, model->hidden_dim);
    layer_norm(hidden2, model->norm1_weight, model->norm1_bias, model->hidden_dim);
    gelu(hidden2, model->hidden_dim);
    
    // Layer 2: Linear (output projection)
    linear(concept_embedding, hidden2,
           model->linear2_weight, model->linear2_bias,
           model->concept_dim, model->hidden_dim);
    
    // L2 normalize
    l2_normalize(concept_embedding, model->concept_dim);
    
    free(hidden);
    free(hidden2);
}
```

## Integration with llama.cpp

To integrate with llama.cpp, add concept predictor support:

### 1. Register Architecture

In `llama.cpp`, add to the architecture enum:

```cpp
enum llm_arch {
    // ... existing architectures ...
    LLM_ARCH_CONCEPT_PREDICTOR,
};

// In architecture name map:
{ LLM_ARCH_CONCEPT_PREDICTOR, "concept_predictor" },
```

### 2. Load Model

```cpp
if (model.arch == LLM_ARCH_CONCEPT_PREDICTOR) {
    // Load concept predictor tensors
    model.concept_pred.text_dim = gguf_get_val_u64(ctx, "concept_predictor.text_dim");
    model.concept_pred.hidden_dim = gguf_get_val_u64(ctx, "concept_predictor.hidden_dim");
    model.concept_pred.concept_dim = gguf_get_val_u64(ctx, "concept_predictor.concept_dim");
    
    // Load weights...
}
```

### 3. Inference API

```cpp
// New API function
void llama_concept_predict(
    struct llama_context* ctx,
    const float* text_embedding,  // From external text encoder
    float* concept_embedding      // Output buffer
);
```

## LM Studio Integration

For LM Studio, create a plugin or use the custom model loader:

```json
{
  "architecture": "concept_predictor",
  "display_name": "Concept-First CodeGen Predictor",
  "requires_external": ["text_encoder", "concept_bank", "code_llm"],
  "pipeline": "concept_first_generation",
  "config": {
    "text_encoder": "Alibaba-NLP/gte-large-en-v1.5",
    "code_llm": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "top_k_concepts": 3,
    "generation_params": {
      "temperature": 0.7,
      "top_p": 0.8,
      "top_k": 20,
      "repetition_penalty": 1.05,
      "max_tokens": 65536
    }
  }
}
```

### Qwen3-Coder Best Practices (from official docs)

- **Temperature**: 0.7 (recommended)
- **Top-P**: 0.8
- **Top-K**: 20
- **Repetition Penalty**: 1.05
- **Max Tokens**: 65536 for most queries
- **Context**: 256K native, up to 1M with YaRN

## File Artifacts

A complete Concept-First deployment includes:

| File | Format | Size | Description |
|------|--------|------|-------------|
| `concept_predictor.gguf` | GGUF | ~4 MB | This MLP model |
| `concept_bank.pt` | PyTorch | ~5 MB | Code embeddings + metadata |
| `gte-large-en-v1.5.gguf` | GGUF | ~600 MB | Text encoder (optional, can use HF) |
| `qwen3-coder-30b-a3b-q4.gguf` | GGUF | ~17 GB | Code generation LLM (MoE, 3B active) |

### Recommended Qwen3-Coder GGUF Sources

| Model | Source | Notes |
|-------|--------|-------|
| Qwen3-Coder-30B-A3B | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | Official unsloth quants |
| Qwen3-Coder-30B-A3B | `lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF` | LM Studio optimized |
| Qwen3-Coder-30B-A3B (1M ctx) | `unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF` | Extended context |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-19 | Initial specification |

## License

This specification is released under MIT License.

## References

- [VL-JEPA Paper (arXiv:2512.10942)](https://arxiv.org/abs/2512.10942) - Original concept prediction approach
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Base format
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Reference implementation target
