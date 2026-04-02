"""
Set-Valued Neural Networks with GPU Acceleration

Implements multi-valued non-linear neural networks where weights are set-valued
{-w, 0, +w}, creating exponentially many paths through the network. Threading
functionals select paths based on hidden state, enabling superposition-like
until measurement (prediction).

Based on: "Set-Valued Matrix Operations: Path Threading Through Admissible Spaces"
Anonymous author, November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class SetValuedLinear(nn.Module):
    """
    Set-valued linear layer where each weight exists in {-w, 0, +w}.
    
    Parameters:
        in_features: Input dimension
        out_features: Output dimension
        branching_factor: Number of values per weight (default 3 for sign-symmetric)
        tie_weights: If True, W_pos = -W_neg (sign-symmetric)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 branching_factor: int = 3, tie_weights: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.branching_factor = branching_factor
        self.tie_weights = tie_weights
        
        # Initialize weight matrices for each branch
        if tie_weights and branching_factor == 3:
            # Sign-symmetric: {-w, 0, +w}
            self.register_parameter('W_base', nn.Parameter(
                torch.randn(in_features, out_features) * 0.01
            ))
            self.register_buffer('W_zero', torch.zeros(in_features, out_features))
        else:
            raise NotImplementedError("Only sign-symmetric (branching=3, tied) supported currently")
        
        # Bias (also set-valued)
        self.register_parameter('b_base', nn.Parameter(
            torch.randn(out_features) * 0.01
        ))
        
    def get_weight_branches(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (W_neg, W_zero, W_pos) weight matrices."""
        W_neg = -self.W_base
        W_zero = self.W_zero
        W_pos = self.W_base
        return W_neg, W_zero, W_pos
    
    def get_bias_branches(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (b_neg, b_zero, b_pos) bias vectors."""
        return -self.b_base, torch.zeros_like(self.b_base), self.b_base
    
    def forward_all_branches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute all branch outputs in parallel.
        
        Args:
            x: Input tensor [batch, in_features]
            
        Returns:
            possibilities: [batch, out_features, 3] tensor where last dim is {neg, zero, pos}
        """
        W_neg, W_zero, W_pos = self.get_weight_branches()
        b_neg, b_zero, b_pos = self.get_bias_branches()
        
        # Compute all three branches in parallel (GPU efficient!)
        out_neg = x @ W_neg + b_neg
        out_zero = x @ W_zero + b_zero
        out_pos = x @ W_pos + b_pos
        
        # Stack: [batch, out_features, 3]
        possibilities = torch.stack([out_neg, out_zero, out_pos], dim=-1)
        
        return possibilities
    
    def forward(self, x: torch.Tensor, threading_pattern: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional threading.
        
        Args:
            x: Input [batch, in_features]
            threading_pattern: [batch, out_features] ∈ {-1, 0, +1} or None
            
        Returns:
            possibilities: [batch, out_features, 3] all possible outputs
            selected: [batch, out_features] threaded output
        """
        possibilities = self.forward_all_branches(x)
        
        if threading_pattern is None:
            # Default: select based on maximum absolute value (strongest signal)
            threading_pattern = self._default_threading(possibilities)
        
        # Select based on pattern
        selected = self._thread_select(possibilities, threading_pattern)
        
        return possibilities, selected
    
    def _default_threading(self, possibilities: torch.Tensor) -> torch.Tensor:
        """Default threading: select max absolute value."""
        # possibilities: [batch, out_features, 3] where dim 2 is {neg, zero, pos}
        abs_vals = possibilities.abs()
        max_idx = abs_vals.argmax(dim=-1)  # [batch, out_features]
        # Convert index to {-1, 0, +1}
        pattern = max_idx - 1  # 0→-1, 1→0, 2→+1
        return pattern
    
    def _thread_select(self, possibilities: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        """
        Select from possibilities based on threading pattern.
        
        Args:
            possibilities: [batch, out_features, 3]
            pattern: [batch, out_features] ∈ {-1, 0, +1}
            
        Returns:
            selected: [batch, out_features]
        """
        # Convert pattern {-1, 0, +1} to indices {0, 1, 2}
        indices = (pattern + 1).long()  # -1→0, 0→1, +1→2
        
        # Gather along last dimension
        batch_size, out_features, _ = possibilities.shape
        indices_expanded = indices.unsqueeze(-1)  # [batch, out_features, 1]
        selected = possibilities.gather(dim=2, index=indices_expanded).squeeze(-1)
        
        return selected


class ThreadingStrategy:
    """Base class for threading strategies."""
    
    def __call__(self, possibilities: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            possibilities: [batch, features, 3]
        Returns:
            pattern: [batch, features] ∈ {-1, 0, +1}
        """
        raise NotImplementedError


class ConfidenceThreading(ThreadingStrategy):
    """
    Thread based on confidence (variance).
    High variance → use 0 (uncertain, defer)
    Low variance → use sign of mean (confident)
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def __call__(self, possibilities: torch.Tensor, **kwargs) -> torch.Tensor:
        # Compute statistics across branches
        mean_val = possibilities.mean(dim=-1)
        var_val = possibilities.var(dim=-1)
        
        # High variance → 0, Low variance → sign of mean
        pattern = torch.where(
            var_val > self.threshold,
            torch.zeros_like(mean_val),
            torch.sign(mean_val)
        ).long()
        
        return pattern


class MaxAbsThreading(ThreadingStrategy):
    """Thread to maximum absolute value (strongest signal)."""
    
    def __call__(self, possibilities: torch.Tensor, **kwargs) -> torch.Tensor:
        abs_vals = possibilities.abs()
        max_idx = abs_vals.argmax(dim=-1)
        pattern = max_idx - 1  # Convert to {-1, 0, +1}
        return pattern


class OptimisticThreading(ThreadingStrategy):
    """Always select positive branch."""
    
    def __call__(self, possibilities: torch.Tensor, **kwargs) -> torch.Tensor:
        batch, features, _ = possibilities.shape
        return torch.ones(batch, features, dtype=torch.long, device=possibilities.device)


class PessimisticThreading(ThreadingStrategy):
    """Always select negative branch."""
    
    def __call__(self, possibilities: torch.Tensor, **kwargs) -> torch.Tensor:
        batch, features, _ = possibilities.shape
        return -torch.ones(batch, features, dtype=torch.long, device=possibilities.device)


class ClassicalThreading(ThreadingStrategy):
    """Always select zero (classical dropout-like)."""
    
    def __call__(self, possibilities: torch.Tensor, **kwargs) -> torch.Tensor:
        batch, features, _ = possibilities.shape
        return torch.zeros(batch, features, dtype=torch.long, device=possibilities.device)


class SetValuedNetwork(nn.Module):
    """
    Multi-layer set-valued neural network.
    
    Architecture:
        Input → SetValuedLinear → Activation → ... → SetValuedLinear → Output
        
    Each layer maintains 3^n paths through the network. Threading functionals
    collapse the superposition at each layer.
    """
    
    def __init__(self, layer_sizes: List[int], threading_strategy: Optional[ThreadingStrategy] = None):
        """
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            threading_strategy: Strategy for path selection (default: MaxAbs)
        """
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.threading_strategy = threading_strategy or MaxAbsThreading()
        
        # Build layers
        self.layers = nn.ModuleList([
            SetValuedLinear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
        
    def forward(self, x: torch.Tensor, return_possibilities: bool = False):
        """
        Forward pass through set-valued network.
        
        Args:
            x: Input [batch, input_size]
            return_possibilities: If True, return all intermediate possibilities
            
        Returns:
            output: [batch, output_size]
            possibilities_list: List of [batch, features, 3] tensors (if return_possibilities=True)
        """
        possibilities_list = []
        
        for i, layer in enumerate(self.layers):
            # Forward through layer
            possibilities, x = layer(x, threading_pattern=None)
            
            # Thread based on strategy
            pattern = self.threading_strategy(possibilities)
            x = layer._thread_select(possibilities, pattern)
            
            # Apply activation (except last layer)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)  # Activation on threaded output
            
            if return_possibilities:
                possibilities_list.append(possibilities)
        
        if return_possibilities:
            return x, possibilities_list
        return x
    
    def forward_ensemble(self, x: torch.Tensor, num_samples: int = 10, use_random: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate ensemble predictions by sampling threading patterns.
        
        Args:
            x: Input [batch, input_size]
            num_samples: Number of threadings to sample
            use_random: If True, use random threading. If False, use self.threading_strategy
            
        Returns:
            mean: [batch, output_size] ensemble mean
            std: [batch, output_size] ensemble std
        """
        outputs = []
        
        for _ in range(num_samples):
            # Forward through network with threading
            curr = x
            for i, layer in enumerate(self.layers):
                possibilities, _ = layer(curr)
                
                if use_random:
                    # Random threading (for exploration)
                    batch, features, branches = possibilities.shape
                    pattern = torch.randint(-1, 2, (batch, features), device=x.device)
                else:
                    # Use model's threading strategy (consistent with training)
                    pattern = self.threading_strategy(possibilities)
                
                curr = layer._thread_select(possibilities, pattern)
                
                if i < len(self.layers) - 1:
                    curr = torch.tanh(curr)
            
            outputs.append(curr)
        
        outputs = torch.stack(outputs)  # [num_samples, batch, output_size]
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        
        return mean, std


def count_paths(network: SetValuedNetwork) -> int:
    """
    Calculate total number of possible paths through network.
    
    For sign-symmetric weights {-w, 0, +w}, each weight contributes branching factor 3.
    Total paths = 3^(total_weights)
    
    Returns:
        Number of distinct paths 
    """
    total_params = sum(p.numel() for p in network.parameters())
    return 3 ** total_params


def estimate_path_count(network: SetValuedNetwork) -> str:
    """Return human-readable path count estimate."""
    total_weights = sum(
        layer.in_features * layer.out_features + layer.out_features
        for layer in network.layers
    )
    
    paths = 3 ** total_weights
    
    # Format in scientific notation
    if paths > 1e100:
        exponent = total_weights * np.log10(3)
        return f"~10^{exponent:.0f} paths"
    else:
        return f"{paths:e} paths"


if __name__ == "__main__":
    print("Set-Valued Neural Network Demo")
    print("=" * 60)
    
    # Create a simple network
    network = SetValuedNetwork(
        layer_sizes=[10, 20, 5],
        threading_strategy=MaxAbsThreading()
    )
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"[+] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[-] Using CPU (GPU not available)")
    
    network = network.to(device)
    
    print(f"\nNetwork Architecture:")
    print(f"  Layers: {network.layer_sizes}")
    print(f"  Threading: {network.threading_strategy.__class__.__name__}")
    print(f"  Total paths: {estimate_path_count(network)}")
    
    # Generate random input
    batch_size = 4
    x = torch.randn(batch_size, 10).to(device)
    
    print(f"\nForward Pass:")
    print(f"  Input shape: {x.shape}")
    
    # Standard forward
    output = network(x)
    print(f"  Output shape: {output.shape}")
    print(f"  Output values:\n{output}")
    
    # Forward with possibilities
    output, possibilities = network(x, return_possibilities=True)
    print(f"\n  Intermediate possibilities:")
    for i, p in enumerate(possibilities):
        print(f"    Layer {i}: {p.shape} -> {p.shape[1] * 3} branches")
    
    # Ensemble forward
    print(f"\nEnsemble Prediction (sampling different paths):")
    mean, std = network.forward_ensemble(x, num_samples=10)
    print(f"  Mean: {mean[0]}")
    print(f"  Std:  {std[0]}")
    
    print("\n" + "=" * 60)
    print("Set-valued neural network demonstration complete!")

