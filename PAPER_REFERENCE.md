# Paper Reference: VL-JEPA

This project is inspired by **VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language** from Meta FAIR.

## Paper Details

- **Title**: VL-JEPA: Joint Embedding Predictive Architecture for Vision-language
- **Authors**: Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Allen Bolourchi, Yann LeCun, Pascale Fung
- **Affiliations**: Meta FAIR, HKUST, Sorbonne Université, NYU
- **arXiv**: [2412.10942](https://arxiv.org/abs/2412.10942)
- **Date**: December 11, 2025
- **License**: CC BY 4.0

## Key Contributions

### 1. Embedding Space Prediction vs Token Space

Traditional VLMs predict tokens autoregressively:
```
L_VLM = D(Ŷ, Y)  # Loss in token space
```

VL-JEPA predicts in embedding space first:
```
L_VL-JEPA = D(Ŝ_Y, S_Y)  # Loss in embedding space
```

**Why this matters**: In token space, semantically equivalent outputs (e.g., "the lamp is off" vs "room goes dark") appear orthogonal. In embedding space, they map to nearby points, simplifying the learning task.

### 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VL-JEPA                               │
├─────────────────────────────────────────────────────────────┤
│  X-Encoder: Vision input → Visual embeddings (S_V)          │
│  Predictor: (S_V, X_Q) → Predicted target embedding (Ŝ_Y)   │
│  Y-Encoder: Text target → Target embedding (S_Y)            │
│  Y-Decoder: Predicted embedding → Text (only at inference)  │
└─────────────────────────────────────────────────────────────┘
```

### 3. Selective Decoding

VL-JEPA can produce continuous streams of semantic embeddings without autoregressive decoding. Decoding happens only when:
- Significant semantic shift is detected
- User requests text output

This enables **2.85× reduction** in decoding operations for streaming video.

### 4. Results

| Task | VL-JEPA | Best Baseline |
|------|---------|---------------|
| Video Classification (8 datasets) | 46.4% | 44.6% (PE-Core-G) |
| Text-to-Video Retrieval (8 datasets) | 58.4% | 58.1% (PE-Core-G) |
| GQA (compositional reasoning) | 60.8% | 59.5% (InternVL-Chat) |

With **50% fewer trainable parameters** and **same training data**.

## How We Apply This to Code Generation

### The Analogy

| VL-JEPA (Vision-Language) | Concept-First (Code) |
|---------------------------|----------------------|
| Visual input X_V | Text query |
| Target text Y | Target code |
| Visual encoder | Text encoder (GTE) |
| Predictor | Concept Predictor (MLP) |
| Y-Encoder | Code encoder (CodeT5+) |
| Y-Decoder | Code LLM (Qwen-Coder) |

### Why It Works for Code

1. **Surface variability**: `arr`, `lst`, `items` are all "the list"
2. **Semantic equivalence**: Many ways to implement the same algorithm
3. **Concept clustering**: Recursive algorithms cluster together

By predicting the **concept** first, we:
- Focus on what the code should *do*, not specific tokens
- Retrieve relevant examples that match the intent
- Guide generation with semantically similar patterns

### Training Objective

We use **InfoNCE loss** (same as CLIP and VL-JEPA):

```python
def info_nce_loss(predicted, target, temperature=0.07):
    logits = (predicted @ target.T) / temperature
    labels = torch.arange(len(predicted))
    
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.T, labels)
    
    return (loss_p2t + loss_t2p) / 2
```

This pulls query embeddings toward their corresponding code embeddings while pushing away from negatives in the batch.

## Citation

```bibtex
@article{chen2024vljepa,
  title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language},
  author={Chen, Delong and Shukor, Mustafa and Moutakanni, Théo and Chung, Willy and 
          Yu, Jade and Kasarla, Tejaswi and Bolourchi, Allen and LeCun, Yann and Fung, Pascale},
  journal={arXiv preprint arXiv:2412.10942},
  year={2024}
}
```

## Related JEPA Work

- **I-JEPA** (Assran et al., 2023): Image JEPA for self-supervised visual learning
- **V-JEPA** (Bardes et al., 2023): Video JEPA for temporal understanding
- **DINO-WM** (Zhou et al., 2025): World models on pre-trained visual features
- **A-JEPA** (Fei et al., 2023): Audio JEPA

The JEPA family represents a shift from reconstruction-based learning (like MAE) to **prediction in latent space**, which:
- Focuses on semantics over surface features
- Enables more efficient learning
- Supports multi-modal alignment
