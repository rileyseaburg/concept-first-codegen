# Concept-First Code Generation

**Predict the "bigger picture" before generating code** - inspired by [VL-JEPA](https://arxiv.org/abs/2512.10942) (Meta FAIR, Dec 2025).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rileyseaburg/concept-first-codegen/blob/main/concept_first_codegen.ipynb)

## The Problem

Traditional autoregressive code models predict tokens one at a time, leading to:
- **Coherence drift** - Starts implementing quicksort, ends up with bubble sort
- **API hallucination** - Invents plausible but fake function names
- **Repetition loops** - `return return return...`

These models optimize for *local* token probability, not *global* semantic coherence.

## The Solution: Concept-First Generation

Inspired by VL-JEPA's approach of predicting in embedding space before decoding, we:

1. **Predict the concept first** - Map the query to a semantic embedding
2. **Retrieve similar patterns** - Find code that matches the predicted concept
3. **Generate with context** - Use retrieved examples to guide generation

```
Query: "Write fibonacci"
         │
         ▼
┌─────────────────────────────┐
│  Concept Predictor (JEPA)   │  "What should this code look like?"
│  Query → Embedding          │
└─────────────────────────────┘
         │
         ▼  [0.23, -0.87, 0.45, ...]
┌─────────────────────────────┐
│  Concept Bank Search        │  "Find similar code patterns"
│  Embedding → Examples       │
└─────────────────────────────┘
         │
         ▼  [fibonacci.py, factorial.py, ...]
┌─────────────────────────────┐
│  Conditioned Generation     │  "Generate with examples as context"
│  Examples + Query → Code    │
└─────────────────────────────┘
         │
         ▼
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Why This Works (VL-JEPA Insight)

From the [VL-JEPA paper](https://arxiv.org/abs/2512.10942):

> "By learning in an abstract representation space, the model focuses on task-relevant semantics while abstracting away surface-level linguistic variability."

In code generation:
- **Surface variability**: `arr`, `lst`, `items`, `data` are all valid variable names
- **Semantic equivalence**: All refer to "the list we're processing"

By predicting in **concept space** first, we capture the *intent* before committing to specific tokens.

### The Math

Traditional VLM loss (token space):
```
L_VLM = D(Ŷ, Y)  where Y is discrete tokens
```

VL-JEPA loss (embedding space):
```
L_JEPA = D(Ŝ_Y, S_Y)  where S is continuous embeddings
```

The embedding space is **smoother** - similar concepts map to nearby points, making learning easier.

## Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| **Concept Encoder** | [CodeT5+ 110M](https://huggingface.co/Salesforce/codet5p-110m-embedding) | Encode code → embeddings |
| **Concept Predictor** | GTE-Large + MLP | Query → predicted embedding |
| **Concept Bank** | FAISS index | Store & retrieve code patterns |
| **Code Generator** | [Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | Generate with context |

### Why Qwen3-Coder?

The latest **Qwen3-Coder** (July 2025) is a Mixture-of-Experts model with:
- **30B total params, 3B active** - Efficient inference
- **128 experts, 8 activated** - Specialized knowledge routing
- **256K context** (1M with YaRN) - Repo-scale understanding
- **Native tool/function calling** - Built for agentic coding

### Training

The Concept Predictor is trained with **InfoNCE loss** (same as CLIP/VL-JEPA):

```python
def info_nce_loss(predicted, target, temperature=0.07):
    """
    Contrastive loss that pulls query embeddings 
    toward their corresponding code embeddings.
    """
    logits = (predicted @ target.T) / temperature
    labels = torch.arange(len(predicted))
    
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.T, labels)
    
    return (loss_p2t + loss_t2p) / 2
```

## Results

### Concept Clustering (t-SNE)

The trained predictor maps similar queries to nearby points:

![Concept Space Visualization](concept_space.png)

- **Red**: Recursion (fibonacci, factorial, tree traversal)
- **Blue**: Sorting (quicksort, mergesort, heapsort)
- **Green**: String operations (palindrome, reverse, substring)
- **Purple**: Data structures (linked list, BST, hash table)

### Retrieval Quality

| Query | Top Match | Similarity |
|-------|-----------|------------|
| "fibonacci recursive" | Fibonacci while loop | 0.678 |
| "check palindrome" | `is_palindrome()` from HumanEval | 0.772 |
| "sort list of numbers" | Sorting with size constraint | 0.744 |
| "binary search" | Binary search on sorted list | 0.532 |

### Code Similarity Matrix

```
               fibonacci  factorial  bubble_sort  binary_search
fibonacci      1.000      0.688      0.284        0.358
factorial      0.688      1.000      0.320        0.406
bubble_sort    0.284      0.320      1.000        0.527
binary_search  0.358      0.406      0.527        1.000
```

Recursive functions cluster together (0.688), iterative algorithms form another cluster (0.527).

## Quick Start

### Run in Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rileyseaburg/concept-first-codegen/blob/main/concept_first_codegen.ipynb)

Requires: A100 (40GB) or T4 (16GB with 4-bit quantization)

### Low VRAM? Use AirLLM (4GB GPU!)

Run Qwen3-Coder-30B on just **4GB VRAM** using [AirLLM](https://github.com/lyogavin/airllm):

```bash
pip install airllm
```

```python
from airllm import AutoModel

# AirLLM loads one layer at a time - fits 70B models on 4GB!
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    compression="4bit"  # 3x faster inference
)
```

See `examples/airllm_low_vram.py` for full integration with concept-first pipeline.

### Local Installation

```bash
git clone https://github.com/rileyseaburg/concept-first-codegen.git
cd concept-first-codegen

pip install -r requirements.txt

# Run the notebook
jupyter notebook concept_first_codegen.ipynb
```

### Python API

```python
from concept_first import ConceptEncoder, ConceptPredictor, ConceptBank, Generator

# Load components
encoder = ConceptEncoder()
predictor = ConceptPredictor.load("concept_predictor.pt")
bank = ConceptBank.load("concept_bank.pt", encoder)
generator = Generator(predictor, bank)

# Generate code
result = generator.generate("implement binary search tree with insert and delete")
print(result["code"])
```

## Datasets

The Concept Bank is built from:

| Dataset | Examples | Source |
|---------|----------|--------|
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | 374 | Google Research |
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | 164 | OpenAI |
| [Evol-Instruct-Code](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 782 | WizardCoder |

Total: **1,320 code concepts** with descriptions

## References

### VL-JEPA (Primary Inspiration)

```bibtex
@article{chen2024vljepa,
  title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language},
  author={Chen, Delong and Shukor, Mustafa and Moutakanni, Théo and Chung, Willy and 
          Yu, Jade and Kasarla, Tejaswi and Bolourchi, Allen and LeCun, Yann and Fung, Pascale},
  journal={arXiv preprint arXiv:2512.10942},
  year={2024}
}
```

**Key insight from VL-JEPA:**
> "VL-JEPA predicts continuous embeddings of the target texts. By learning in an abstract representation space, the model focuses on task-relevant semantics while abstracting away surface-level linguistic variability."

### Related Work

- **I-JEPA** ([Assran et al., 2023](https://arxiv.org/abs/2301.08243)) - Image JEPA
- **V-JEPA** ([Bardes et al., 2023](https://arxiv.org/abs/2404.08471)) - Video JEPA  
- **CLIP** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) - Contrastive Language-Image Pre-training
- **CodeT5+** ([Wang et al., 2023](https://arxiv.org/abs/2305.07922)) - Code embedding model

## Project Structure

```
concept-first-codegen/
├── README.md                    # This file
├── concept_first_codegen.ipynb  # Main notebook
├── requirements.txt             # Dependencies
├── concept_first/               # Python package
│   ├── __init__.py
│   ├── encoder.py               # ConceptEncoder
│   ├── predictor.py             # ConceptPredictor  
│   ├── bank.py                  # ConceptBank
│   └── generator.py             # ConceptFirstGenerator
├── checkpoints/                 # Saved models
│   ├── concept_predictor.pt
│   └── concept_bank.pt
└── examples/                    # Usage examples
    └── basic_usage.py
```

## Future Work

1. **Hierarchical Concepts** - Predict at multiple levels (program → function → block)
2. **Streaming Generation** - Update concept embedding as code is generated
3. **Distillation** - Train small models to predict concepts directly
4. **Benchmarks** - HumanEval/MBPP pass@1 comparison

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use this work, please cite:

```bibtex
@software{seaburg2026conceptfirst,
  author = {Seaburg, Riley},
  title = {Concept-First Code Generation},
  year = {2026},
  url = {https://github.com/rileyseaburg/concept-first-codegen}
}
```

And the VL-JEPA paper that inspired this approach:

```bibtex
@article{chen2024vljepa,
  title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language},
  author={Chen, Delong and Shukor, Mustafa and Moutakanni, Théo and Chung, Willy and 
          Yu, Jade and Kasarla, Tejaswi and Bolourchi, Allen and LeCun, Yann and Fung, Pascale},
  journal={arXiv preprint arXiv:2512.10942},
  year={2024}
}
```
