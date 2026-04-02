# Legacy Hybrid Architecture Overview

## Purpose

This directory preserves the main experimental line that preceded the current Eidos architecture. The goal is reproducibility of the older hybrid path-preserving work without carrying forward every internal debugging artifact from the original archive.

## Core Ideas

The legacy line explored the following claims:

- weights can be treated as set-valued branches `{-W, 0, +W}`
- this induces a growing path space through the network
- naive averaging across all paths is destructive
- learned scoring and pruning can recover useful paths
- preserving paths across layers is materially different from collapsing at each layer

## Main Implementations

### `set_valued_nn.py`

Early fully connected set-valued networks and threading strategies for small-scale experiments.

### `set_valued_pathpres.py`

General path-bundle and path-preserving layer implementation used by the learned-sigma branch.

### `set_valued_learned_sigma.py`

Legacy attempt to learn a state-dependent path-selection signal (`sigma`) over preserved paths.

### `set_valued_cnn.py`

Set-valued convolutional layers plus several CNN baselines, including layer-local collapse and path-quality variants.

### `set_valued_cnn_hybrid.py`

Hybrid architecture combining layer-local convolutional handling with path-quality fully connected stages.

### `set_valued_cnn_pathbundle.py`

True path-preserving CNN variant that carries multiple spatial paths across layers rather than collapsing immediately.

## Durable Observations

The main observations from this line that still matter historically are:

- selection is necessary because naive path averaging destroys signal
- path preservation across depth behaves differently from layer-local collapse
- path-granularity mismatches matter, especially for spatial data
- path diversity suggested an early route toward uncertainty and confidence diagnostics

## Boundaries

This snapshot should not be read as the current architecture or as the final theoretical framing. In particular:

- it predates the native geometric matcher line
- it predates ProbableCollapse
- it predates the current observer/collapse formulation
- some historical claims in the original archive were stronger than the evidence justified

Accordingly, this release keeps the code and the durable architectural observations while removing the more transient status language from the old archive.
