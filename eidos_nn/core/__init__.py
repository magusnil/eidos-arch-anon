"""
eidos Neural Network Core Module

This module provides the foundational components for Pure eidos neural networks:
- State: Stateful computation tracking with carry flags (NeuralState, StateManager)
- PathBundle: Multi-path tensor data structure for set-valued operations
- Axioms: R1-R4 axiom implementations bridged to neural operations
"""

from .state import NeuralState, StateManager, get_default_state_manager
from .path_bundle import (
    PathBundle,
    apply_per_path,
    concat_bundles
)
__all__ = [
    # Path bundles
    'PathBundle',
    'apply_per_path',
    'concat_bundles',
]

__version__ = '0.1.0'