"""
Path-Preserving Set-Valued Neural Network Architecture

Maintains coherent paths through all layers instead of collapsing at each layer.
"""
import torch
import torch.nn as nn


class PathBundle:
    """
    Represents multiple paths simultaneously.
    
    Shape: [batch, features, num_paths]
    Each slice [..., :, path_i] is an independent execution path.
    """
    
    def __init__(self, data):
        # data: [batch, features, num_paths]
        self.data = data
        self.num_paths = data.shape[-1]
    
    def apply_activation(self, activation_fn):
        """Apply activation independently to each path."""
        return PathBundle(activation_fn(self.data))
    
    def thread(self, strategy):
        """Collapse paths using threading strategy."""
        # strategy selects from dim=-1
        return strategy(self.data)


class PathPreservingLayer(nn.Module):
    """
    Layer that operates on PathBundles, expanding path space.
    
    Input:  [batch, in_features, num_paths_in]
    Output: [batch, out_features, num_paths_in * 3]
    
    Each input path spawns 3 output paths {-w, 0, +w}.
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Single base weight (sign-symmetric)
        self.W_base = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.b_base = nn.Parameter(torch.randn(out_features) * 0.01)
    
    def forward(self, path_bundle: PathBundle) -> PathBundle:
        """
        Expand each input path into 3 output paths (GPU-vectorized).
        
        Args:
            path_bundle: [batch, in_features, num_paths_in]
        
        Returns:
            [batch, out_features, num_paths_in * 3]
        """
        # path_bundle.data: [batch, in_features, num_paths_in]
        batch, in_feat, num_paths = path_bundle.data.shape
        
        # Transpose for batch matrix multiply: [batch, num_paths, in_features]
        paths_t = path_bundle.data.transpose(1, 2)
        
        # Compute all three branches in parallel
        out_neg = paths_t @ (-self.W_base) + (-self.b_base)  # [batch, num_paths, out_features]
        out_zero = torch.zeros_like(out_neg)
        out_pos = paths_t @ self.W_base + self.b_base
        
        # Stack branches: [batch, num_paths, out_features, 3]
        branches = torch.stack([out_neg, out_zero, out_pos], dim=-1)
        
        # Reshape to [batch, out_features, num_paths * 3]
        all_paths = branches.transpose(1, 2).reshape(batch, self.out_features, -1)
        
        return PathBundle(all_paths)


class PathPreservingNetwork(nn.Module):
    """
    Network that maintains path identity through all layers.
    
    Architecture:
        Input [batch, features] 
        → PathBundle [batch, features, 1]           # Start with single path
        → Layer1 [batch, h1, 3]                     # Expand to 3 paths
        → Activation (per-path)
        → Layer2 [batch, h2, 9]                     # Expand to 9 paths
        → Activation (per-path)
        → Layer3 [batch, out, 27]                   # Expand to 27 paths
        → Thread (collapse to single output)
    """
    
    def __init__(self, layer_sizes, max_paths=1000):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.max_paths = max_paths  # Computational limit
        
        self.layers = nn.ModuleList([
            PathPreservingLayer(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
    
    def forward(self, x, threading_strategy=None):
        """
        Args:
            x: [batch, input_size]
            threading_strategy: How to collapse paths at end
        
        Returns:
            output: [batch, output_size]
            all_paths: [batch, output_size, num_final_paths] (for analysis)
        """
        batch_size = x.shape[0]
        
        # Initialize with single path
        paths = PathBundle(x.unsqueeze(-1))  # [batch, input_size, 1]
        
        for layer_idx, layer in enumerate(self.layers):
            # Expand paths
            paths = layer(paths)
            
            # Apply activation (except last layer)
            if layer_idx < len(self.layers) - 1:
                paths = paths.apply_activation(torch.tanh)
            
            # Path pruning if too many
            if paths.num_paths > self.max_paths:
                paths = self._prune_paths(paths, keep=self.max_paths)
        
        # Collapse paths using threading
        if threading_strategy is None:
            threading_strategy = 'max_abs'
        
        output = self._collapse_paths(paths.data, threading_strategy)
        
        return output, paths.data
    
    def _collapse_paths(self, path_data: torch.Tensor, strategy: str) -> torch.Tensor:
        """
        Collapse path bundle to single output.
        
        Args:
            path_data: [batch, features, num_paths]
            strategy: 'max_abs', 'mean', 'strongest', 'random'
        
        Returns:
            [batch, features]
        """
        if strategy == 'max_abs':
            # Select path with maximum absolute value per feature
            abs_vals = path_data.abs()
            max_idx = abs_vals.argmax(dim=-1, keepdim=True)
            output = path_data.gather(dim=-1, index=max_idx).squeeze(-1)
        
        elif strategy == 'mean':
            # Average all paths
            output = path_data.mean(dim=-1)
        
        elif strategy == 'strongest':
            # Select path with max L2 norm across features
            path_norms = path_data.norm(dim=1)  # [batch, num_paths]
            strongest_idx = path_norms.argmax(dim=-1)  # [batch]
            
            # Gather strongest path
            batch, features, num_paths = path_data.shape
            strongest_idx = strongest_idx.view(batch, 1, 1).expand(-1, features, -1)
            output = path_data.gather(dim=-1, index=strongest_idx).squeeze(-1)
        
        elif strategy == 'random':
            # Random path selection
            batch, features, num_paths = path_data.shape
            random_idx = torch.randint(0, num_paths, (batch,), device=path_data.device)
            random_idx = random_idx.view(batch, 1, 1).expand(-1, features, -1)
            output = path_data.gather(dim=-1, index=random_idx).squeeze(-1)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return output
    
    def _prune_paths(self, paths: PathBundle, keep: int) -> PathBundle:
        """
        Prune paths to keep only top-k by some criterion.
        
        Options:
        - Keep strongest (by L2 norm)
        - Keep most diverse (by variance)
        - Random sample
        """
        # For now: keep strongest paths by L2 norm
        path_strengths = paths.data.norm(dim=1)  # [batch, num_paths]
        _, top_indices = path_strengths.topk(keep, dim=-1)
        
        # Gather top paths
        batch_size, features, _ = paths.data.shape
        top_indices_expanded = top_indices.unsqueeze(1).expand(-1, features, -1)
        pruned = paths.data.gather(dim=-1, index=top_indices_expanded)
        
        return PathBundle(pruned)