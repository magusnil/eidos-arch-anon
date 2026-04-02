"""
Set-Valued Neural Network with Learned Threading State σ

Implements the mathematical framework where threading functional T_σ
depends on learned state σ that determines path selection.
"""
import torch
import torch.nn as nn
from set_valued_pathpres import PathBundle, PathPreservingLayer


class LearnedSigmaPathNetwork(nn.Module):
    """
    Path-preserving network where σ is learned from input.
    
    Key insight: σ(x) determines which paths are good for input x.
    The network learns both the paths AND the selection strategy.
    """
    
    def __init__(self, layer_sizes, sigma_hidden=64, max_paths=100):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.max_paths = max_paths
        
        # Build path-preserving layers
        self.layers = nn.ModuleList([
            PathPreservingLayer(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
        
        # LEARN σ: Maps input → selection state
        self.sigma_network = nn.Sequential(
            nn.Linear(layer_sizes[0], sigma_hidden),
            nn.ReLU(),
            nn.Linear(sigma_hidden, sigma_hidden // 2),
            nn.ReLU(),
            nn.Linear(sigma_hidden // 2, 16),
            nn.Tanh()  # σ ∈ [-1, 1]^16 (multi-dimensional state)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_size]
        
        Returns:
            output: [batch, output_size]
            sigma: [batch, sigma_dim] (for analysis)
        """
        batch_size = x.shape[0]
        
        # Compute σ from input
        sigma = self.sigma_network(x)  # [batch, 16]
        
        # Initialize paths
        paths = PathBundle(x.unsqueeze(-1))  # [batch, input_size, 1]
        
        # Forward through layers with σ-guided selection
        for layer_idx, layer in enumerate(self.layers):
            # Expand paths
            paths = layer(paths)
            
            # Prune using σ-weighted quality
            if paths.num_paths > self.max_paths:
                paths = self._sigma_weighted_prune(paths, sigma, keep=self.max_paths)
            
            # Activation (except last layer)
            if layer_idx < len(self.layers) - 1:
                paths = paths.apply_activation(torch.tanh)
        
        # Final collapse using σ
        output = self._sigma_weighted_collapse(paths, sigma)
        
        return output, sigma
    
    def _sigma_weighted_prune(self, paths: PathBundle, sigma: torch.Tensor, keep: int) -> PathBundle:
        """
        Prune paths based on σ-weighted quality.
        
        σ determines what "quality" means for this input.
        Different σ → different quality metrics → different paths kept.
        """
        # paths.data: [batch, features, num_paths]
        # sigma: [batch, sigma_dim]
        
        batch_size, features, num_paths = paths.data.shape
        
        # Compute path scores based on σ
        # Use σ to weight different quality metrics
        sigma_expanded = sigma.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, sigma_dim]
        
        # Quality metric 1: Path strength (L2 norm)
        path_strength = paths.data.norm(dim=1)  # [batch, num_paths]
        
        # Quality metric 2: Path sparsity
        path_sparsity = (paths.data.abs() < 0.1).float().mean(dim=1)  # [batch, num_paths]
        
        # Quality metric 3: Path diversity (variance across features)
        path_diversity = paths.data.var(dim=1)  # [batch, num_paths]
        
        # Combine metrics using σ weights
        # σ[0:5] controls strength weight
        # σ[5:10] controls sparsity weight
        # σ[10:15] controls diversity weight
        strength_weight = sigma[:, 0:5].mean(dim=1, keepdim=True)  # [batch, 1]
        sparsity_weight = sigma[:, 5:10].mean(dim=1, keepdim=True)
        diversity_weight = sigma[:, 10:15].mean(dim=1, keepdim=True)
        
        # Normalize weights
        total_weight = strength_weight.abs() + sparsity_weight.abs() + diversity_weight.abs() + 1e-8
        strength_weight = strength_weight / total_weight
        sparsity_weight = sparsity_weight / total_weight
        diversity_weight = diversity_weight / total_weight
        
        # Compute overall quality score
        path_quality = (
            strength_weight * path_strength + 
            sparsity_weight * path_sparsity + 
            diversity_weight * path_diversity
        )  # [batch, num_paths]
        
        # Keep top-k paths by σ-weighted quality
        _, top_indices = path_quality.topk(keep, dim=-1)
        
        # Gather top paths
        top_indices_expanded = top_indices.unsqueeze(1).expand(-1, features, -1)
        pruned = paths.data.gather(dim=-1, index=top_indices_expanded)
        
        return PathBundle(pruned)
    
    def _sigma_weighted_collapse(self, paths: PathBundle, sigma: torch.Tensor) -> torch.Tensor:
        """
        Collapse paths using σ-weighted combination.
        
        σ determines how to combine multiple paths into final output.
        This is the threading functional T_σ.
        """
        # paths.data: [batch, features, num_paths]
        # sigma: [batch, sigma_dim]
        
        batch_size, features, num_paths = paths.data.shape
        
        # Compute path weights from σ
        # Use last dimension of σ to control collapse strategy
        collapse_mode = sigma[:, -1]  # [batch]
        
        # Mode 1: Average (when σ[-1] ≈ -1)
        # Mode 2: Max-abs (when σ[-1] ≈ +1)
        # Interpolate between them
        
        # Average collapse
        avg_output = paths.data.mean(dim=-1)  # [batch, features]
        
        # Max-abs collapse
        abs_vals = paths.data.abs()
        max_idx = abs_vals.argmax(dim=-1, keepdim=True)
        maxabs_output = paths.data.gather(dim=-1, index=max_idx).squeeze(-1)
        
        # Interpolate based on σ[-1]
        # σ[-1] = -1 → all average
        # σ[-1] = +1 → all max-abs
        alpha = (collapse_mode.unsqueeze(1) + 1) / 2  # [batch, 1], range [0, 1]
        
        output = (1 - alpha) * avg_output + alpha * maxabs_output
        
        return output


class PathQualityNetwork(nn.Module):
    """
    Alternative: Explicitly learn path quality scores.
    
    Instead of implicit σ weighting, directly score each path.
    """
    
    def __init__(self, layer_sizes, max_paths=100):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.max_paths = max_paths
        
        # Path layers
        self.layers = nn.ModuleList([
            PathPreservingLayer(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
        
        # Path quality scorers (one per layer)
        self.quality_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i+1], 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for i in range(len(layer_sizes) - 1)
        ])
    
    def forward(self, x):
        """Forward with explicit path quality scoring."""
        batch_size = x.shape[0]
        paths = PathBundle(x.unsqueeze(-1))
        
        for layer_idx, (layer, scorer) in enumerate(zip(self.layers, self.quality_scorers)):
            # Expand paths
            paths = layer(paths)
            
            # Score each path
            if paths.num_paths > self.max_paths:
                paths = self._quality_prune(paths, scorer, keep=self.max_paths)
            
            # Activation
            if layer_idx < len(self.layers) - 1:
                paths = paths.apply_activation(torch.tanh)
        
        # Final: weighted average by quality
        output = self._quality_weighted_output(paths, self.quality_scorers[-1])
        
        # Ensure correct output shape [batch, features]
        assert output.shape[0] == batch_size, f"Batch dim mismatch: got {output.shape}, expected [{batch_size}, features]"
        
        # Return tuple to match other models' interface
        return output, None
    
    def _quality_prune(self, paths: PathBundle, scorer: nn.Module, keep: int) -> PathBundle:
        """Prune by learned quality scores."""
        # paths.data: [batch, features, num_paths]
        batch_size, features, num_paths = paths.data.shape
        
        # Score each path
        paths_t = paths.data.transpose(1, 2)  # [batch, num_paths, features]
        scores = scorer(paths_t).squeeze(-1)  # [batch, num_paths]
        
        # Keep top-k
        _, top_indices = scores.topk(keep, dim=-1)
        top_indices_expanded = top_indices.unsqueeze(1).expand(-1, features, -1)
        pruned = paths.data.gather(dim=-1, index=top_indices_expanded)
        
        return PathBundle(pruned)
    
    def _quality_weighted_output(self, paths: PathBundle, scorer: nn.Module) -> torch.Tensor:
        """Output as quality-weighted average of paths."""
        batch_size, features, num_paths = paths.data.shape
        
        # Score paths
        paths_t = paths.data.transpose(1, 2)  # [batch, num_paths, features]
        scores = scorer(paths_t).squeeze(-1)  # [batch, num_paths]
        
        # Softmax to get weights
        weights = torch.softmax(scores, dim=-1)  # [batch, num_paths]
        
        # Weighted sum
        weights_expanded = weights.unsqueeze(1)  # [batch, 1, num_paths]
        output = (paths.data * weights_expanded).sum(dim=-1)  # [batch, features]
        
        return output


def estimate_path_count_learned(network: LearnedSigmaPathNetwork) -> str:
    """Estimate number of paths (same as before)."""
    total_paths = 1
    for layer in network.layers:
        total_paths *= 3
    
    if total_paths > 1000:
        exp = len(str(total_paths)) - 1
        return f"~10^{exp}"
    return str(total_paths)

