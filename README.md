# Eidos Neural Architecture

**Measurement-Driven Geometric Deep Learning**

This repository is a private reviewer/evaluation release, not an open-source release. Use is governed by the repository `LICENSE`.

Eidos replaces the standard deep learning primitives (nn.Linear, LayerNorm, pointwise activations) with geometric operations derived from a set of algebraic axioms (R1–R6). The core non-linearity is *discrete path selection*: multiple candidate transformations are generated in parallel and an Observer module selects a single winner via hard Gumbel-Softmax collapse.

## Quick Start

```bash
pip install torch torchvision tqdm scipy numpy
```

### Classification (MNIST)
```bash
python examples/train_mnist_fast.py
```

### Classification (CIFAR-10)
```bash
python examples/train_cifar_fast.py
```

### Sentiment Analysis (IMDB)
```bash
python examples/train_imdb_fast.py --d-model 128 --epochs 3
```

### Autoregressive Generation
```bash
python examples/train_autoregressive.py
```

### Paper-Specific Reproduction
ICML 2026 Dataset 1 scripts are kept separate from the generic examples:

```bash
python papers/icml2026/train_dataset1_v7_mlp_baselines.py --help
python papers/icml2026/train_dataset1_v7_table3.py --help
python papers/icml2026/train_dataset1_v7_table5.py --help
```

## Architecture Overview

| Component | Replaces | Description |
|-----------|----------|-------------|
| `eidosTransform` | `nn.Linear` | Givens rotations (R5) + log-scale (R6) + residual identity (R1) |
| `ModularPhaseNorm` | `nn.LayerNorm` | RMS normalization with modular-cycle phase modulation |
| `ProbableCollapseLayer` | Soft attention | Hard Gumbel-Softmax selection from 9 candidate paths |
| `MobiusCollapseLayer` | — | Fixed-point iterative collapse with vertical coherence for autoregression |
| `FractalOptimizer` | `Adam` | 3-band gradient decomposition (coarse/triadic/fine) |
| `SetValuedAttention` | `nn.MultiheadAttention` | Q/K/V via eidosTransform with float64 precision softmax |
| `HierarchicalPathScorer` | — | Team-decomposed quality scoring (Brooks's Law) |

## Project Structure

```
eidos_nn/
├── layers/
│   ├── eidos_transform.py          # Core geometric projection (replaces nn.Linear)
│   ├── convolution.py              # Set-valued spatial convolution
│   ├── set_valued_attention.py     # Geometric Q/K/V attention
│   ├── neighbor_mixer.py           # Local [Left, Self, Right] context fusion
│   ├── hierarchical_scorer.py      # Team-based path quality scoring
│   ├── form_first_color.py         # RGB/CMYK geometric color encoding
│   ├── form_space_mapper.py        # Multi-scale contrastive form-space mapping
│   ├── ir_positional_encoding.py   # Dimensional-regime hierarchical positions
│   └── true_eidos_ffn.py           # Set-valued feed-forward network
├── models/
│   ├── eidos_measurement_driven.py # ProbableCollapseLayer (classification)
│   └── mobius_collapse_layer.py    # MobiusCollapseLayer (autoregression)
├── optim/
│   └── fractal_optimizer.py        # Multi-frequency gradient optimizer
└── utils/
    ├── modular_phase_norm.py       # Geometric normalization
    ├── measure_structural_tension.py
    ├── certainty_validity.py
    ├── imdb_utils.py
    └── logger.py

examples/
├── train_mnist_fast.py
├── train_cifar_fast.py
├── train_imdb_fast.py
└── train_autoregressive.py

hybrid-old-architecture/
├── README.md
├── core/                           # Legacy set-valued / path-preserving precursor
├── demo/                           # Historical MNIST and CNN validation scripts
└── doc/                            # Cleaned overview of the precursor experiments

papers/
└── icml2026/
    ├── README.md
    ├── train_dataset1_v7_mlp_baselines.py
    ├── train_dataset1_v7_table3.py
    ├── train_dataset1_v7_table5.py
    └── train_dataset1_v7_pathbundle.py
```

## Key Results

### Dimensional Invariance (IMDB, 1 Epoch)

| Dimension | Parameters | Train Loss | Train Acc | Test Loss | Test Acc | Train Time | Eval Time |
|-----------|-----------|------------|-----------|-----------|----------|------------|-----------|
| d=128     | 1,542,022 | 0.5529     | 69.59%    | 0.5002    | **74.78%** | ~2 min   | ~49 sec   |
| d=576     | 10,938,710| 0.5167     | 72.24%    | 0.5044    | **77.22%** | ~7 min   | ~3 min    |

> **Platonic Spike**: In both runs, test accuracy exceeds training accuracy at Epoch 1.
> This indicates the architecture is discovering generalizable rules rather than
> memorizing training statistics—a structural signature of discrete path selection.

### Dimensional Scaling Trade-offs

Lower dimensions (e.g. `d=32`, ~338K params and `d=128`, ~1.5M params) converge *faster* per-epoch but exhibit **benign overfitting** earlier—the Platonic Spike appears sooner and the train/test gap widens more quickly. Higher dimensions (e.g. `d=576`) sustain the Platonic Spike across more epochs because the larger geometric manifold better resists memorization pressure.

| Behavior | Low Dim (d=32) | High Dim (d=576) |
|----------|----------------|------------------|
| Convergence speed | Fast | Slower |
| Platonic Spike duration | Brief (1–2 epochs) | Extended (3+ epochs) |
| Benign overfitting onset | Early | Late |
| Parameter efficiency | High | Lower |

Choose lower dims for rapid prototyping and resource-constrained settings; choose higher dims when maximal generalization matters.

### Benchmark Summary

| Benchmark | Accuracy | Parameters |
|-----------|----------|------------|
| MNIST     | ~99%     | ~1.2M      |
| CIFAR-10  | ~77%     | ~8M        |

## Structural Positioning

The table below summarizes how Eidos differs from adjacent paradigms across three practical axes: where geometry is located, what the core primitive is, and where the nonlinearity originates.

| Paradigm | Where is the geometry? | Core primitive | Nonlinearity source |
|----------|------------------------|----------------|---------------------|
| Standard Transformer | Absent | Matrix multiply | Pointwise activation |
| Geometric DL | In the data domain | Equivariant matmul | Pointwise activation |
| PINNs | In the loss function | Matrix multiply | Pointwise activation |
| SNNs | In temporal dynamics | Spike integration | Threshold firing |
| Lie Equivariant | Constrains the weights | Constrained matmul | Pointwise activation |
| Mixture of Experts | Absent | Pre-routed matmul | Pointwise activation |
| **Eidos** | **In the computation** | **Rotation + scaling** | **Set-valued collapse** |

In practical terms:

- Eidos does not place geometry in the data domain or only in the loss; it places geometry directly in the layer operations.
- The core transform is not a standard affine projection but a rotation-and-scaling stack (`eidosTransform`).
- The main nonlinearity comes from set-valued branching and hard path collapse rather than pointwise activations.

## Fractal Optimizer

`FractalOptimizer` replaces a single global update scale with a three-band decomposition aligned to the architecture's branching structure.

| Band | Exponent | Role |
|------|----------|------|
| Coarse | 0.5 | Large-scale stable anchor updates |
| Triadic | 0.3 | Branching-scale updates aligned to `{-W, 0, +W}` |
| Fine | 0.2 | Small-scale path diversity refinement |

The effective learning rates are:

```text
lr_coarse  = base_lr * (batch_scale ^ 0.5)
lr_triadic = base_lr * (batch_scale ^ 0.3)
lr_fine    = base_lr * (batch_scale ^ 0.2)
```

where `batch_scale = actual_batch_size / 32`.

The companion `FractalScheduler` provides short warmup and post-warmup restoration of the full bandwise rates.

## Ambiguity Ceiling

Across several standard datasets, the architecture repeatedly plateaus near the low-80% range when the label structure itself is ambiguous. On cleaner structural subsets, the same system reaches much higher accuracy and often exhibits a stronger early validation-leading regime.

| Dataset | Condition | Early gap | Peak accuracy |
|---------|-----------|-----------|---------------|
| MNIST | Full | +13% | 99.86% |
| Fashion-MNIST | Full | ~0% | 83% |
| Fashion-MNIST | Clean subset | +14.69% | 97% |
| EMNIST | Digits only | +3.94% | 99.59% |
| IMDB | Full | ~0% | 83% |
| IMDB | Strong sentiment only | +7.24% | 87% |

The practical read is that Eidos does not simply "fail at 83%": it performs strongly on the clean structural portion of the task and becomes conservative on the ambiguous remainder.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision, tqdm, scipy, numpy

## Citation

Formal citation information is intentionally withheld in this reviewer/evaluation release.
