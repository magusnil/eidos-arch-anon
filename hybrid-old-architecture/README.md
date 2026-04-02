# Hybrid Old Architecture

This folder preserves an academically sanitized snapshot of the pre-Eidos hybrid architecture line. It is included for reproducibility and historical context, not as the recommended starting point for new work.

The package captures the period where the architecture program centered on:

- set-valued weights with branches `{-W, 0, +W}`
- explicit path growth and pruning
- hybrid conv/fully-connected designs
- early path-preserving experiments before the later native geometric Eidos stack

## What Is Included

- `core/`
  Early set-valued and path-preserving implementations.
- `demo/`
  Historical MNIST and CIFAR-style validation scripts.
- `doc/`
  Cleaned overview notes describing the experiment line and its main outcomes.
- `requirements.txt`
  Minimal dependency list for the legacy code.

## What Is Not Included

- transient debugging notes
- internal status dashboards
- stale geometric-extension summaries
- claims that depended on unfinished optimization work

## Historical Position

This snapshot is best understood as a precursor to the later Eidos architecture:

1. set-valued branches exposed explicit path multiplicity
2. naive path averaging failed
3. learned path scoring and pruning became necessary
4. path preservation across layers emerged as the central mechanism
5. later work moved from this hybrid line into the native geometric Eidos formulation

## Directory Layout

```text
hybrid-old-architecture/
├── core/
│   ├── set_valued_nn.py
│   ├── set_valued_pathpres.py
│   ├── set_valued_learned_sigma.py
│   ├── set_valued_cnn.py
│   ├── set_valued_cnn_hybrid.py
│   └── set_valued_cnn_pathbundle.py
├── demo/
│   ├── set_valued_mnist_demo.py
│   └── final_cnn_validation.py
├── doc/
│   └── OVERVIEW.md
└── requirements.txt
```

## Running The Demos

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the MNIST demo:

```bash
python demo/set_valued_mnist_demo.py
```

Run the CIFAR-style historical validation:

```bash
python demo/final_cnn_validation.py
```

The demos now set up their import path relative to this folder so the snapshot can run without depending on the original archive layout.

## Notes On Reproducibility

- This package intentionally preserves the older hybrid/path-preserving line rather than rewriting it in current Eidos terms.
- `set_valued_pathpres.py` is included from the original `uof_nn` archive because it was a real dependency of the learned-sigma branch and is required for faithful reproduction.
- The code is preserved with only minimal import-path cleanup and documentation cleanup.

## Use In The Public Release

Use the main `public_release/README.md` and the `eidos_nn/` package for the current architecture.

Use this folder when you want:

- the historical hybrid precursor
- early path-preserving experiments
- an academically cleaned legacy reference for comparison
