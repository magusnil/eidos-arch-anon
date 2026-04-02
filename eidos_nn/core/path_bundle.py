"""
PathBundle: Core data structure for set-valued neural operations.

A PathBundle represents multiple execution paths through a neural network,
where each path corresponds to a different branch of set-valued operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Callable


class PathBundle:
    """
    A PathBundle is a tensor of shape [batch, features, num_paths] where
    each slice [:, :, i] represents the i-th execution path.
    
    This is the fundamental data structure for Pure eidos neural networks,
    allowing operations to maintain and manipulate multiple execution paths
    simultaneously.
    """
    
    def __init__(
        self, 
        data: torch.Tensor, 
        num_paths: Optional[int] = None,
        quality_scores: Optional[torch.Tensor] = None
    ):
        """
        Initialize a PathBundle.
        
        Args:
            data: Tensor of shape [batch, features] or [batch, features, paths]
            num_paths: If data is 2D, expand to this many paths (all identical)
            quality_scores: Optional tensor of shape [batch, paths] with path qualities
        """
        if data.dim() == 2 and num_paths is not None:
            # Expand 2D tensor to 3D with identical paths
            batch, features = data.shape
            data = data.unsqueeze(-1).expand(batch, features, num_paths)
        
        if data.dim() != 3:
            raise ValueError(f"PathBundle data must be 3D, got shape {data.shape}")
        
        self.data = data
        self.batch_size, self.features, self.num_paths = data.shape
        
        # Quality scores default to uniform if not provided
        if quality_scores is None:
            self.quality_scores = torch.ones(
                self.batch_size, self.num_paths, 
                device=data.device, dtype=data.dtype
            ) / self.num_paths
        else:
            self.quality_scores = quality_scores
    
    @property
    def shape(self):
        """Return shape as (batch, features, paths)"""
        return self.data.shape
    
    @property
    def device(self):
        """Return device of underlying tensor"""
        return self.data.device
    
    @property
    def dtype(self):
        """Return dtype of underlying tensor"""
        return self.data.dtype
    
    def __getitem__(self, idx):
        """Allow indexing into paths: bundle[i] returns path i"""
        if isinstance(idx, int):
            return self.data[:, :, idx]
        elif isinstance(idx, slice):
            return PathBundle(
                self.data[:, :, idx],
                quality_scores=self.quality_scores[:, idx]
            )
        else:
            raise TypeError(f"PathBundle indices must be int or slice, got {type(idx)}")
    
    def get_path(self, path_idx: int) -> torch.Tensor:
        """Get a specific path as a 2D tensor [batch, features]"""
        return self.data[:, :, path_idx]
    
    def set_quality_scores(self, scores: torch.Tensor):
        """Set quality scores for all paths"""
        if scores.shape != (self.batch_size, self.num_paths):
            raise ValueError(
                f"Quality scores shape {scores.shape} doesn't match "
                f"expected ({self.batch_size}, {self.num_paths})"
            )
        self.quality_scores = scores
    
    def prune(self, k: int) -> 'PathBundle':
        """
        Keep only top-k paths by quality score.
        
        Args:
            k: Number of paths to keep
            
        Returns:
            New PathBundle with k paths
        """
        if k >= self.num_paths:
            return self
        
        # Get top-k indices for each batch element
        top_k_values, top_k_indices = torch.topk(self.quality_scores, k, dim=1)
        
        # Gather paths
        # top_k_indices shape: [batch, k]
        # We need to expand it to [batch, features, k] for gathering
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        pruned_data = torch.zeros(
            self.batch_size, self.features, k,
            device=self.device, dtype=self.dtype
        )
        
        for b in range(self.batch_size):
            for p_new, p_old in enumerate(top_k_indices[b]):
                pruned_data[b, :, p_new] = self.data[b, :, p_old]
        
        return PathBundle(pruned_data, quality_scores=top_k_values)
    
    def collapse(self, strategy: str = 'weighted') -> torch.Tensor:
        """
        Collapse PathBundle to single tensor [batch, features].
        
        Args:
            strategy: How to collapse paths
                - 'weighted': Quality-weighted average
                - 'mean': Simple average
                - 'max': Take maximum across paths
                - 'best': Take path with highest quality
        
        Returns:
            Tensor of shape [batch, features]
        """
        if strategy == 'weighted':
            # Weighted average: sum(path_i * quality_i)
            # quality_scores: [batch, paths] -> [batch, 1, paths]
            weights = self.quality_scores.unsqueeze(1)  
            # data: [batch, features, paths]
            # weights: [batch, 1, paths]
            # Result: [batch, features, paths] -> [batch, features]
            return (self.data * weights).sum(dim=-1)
        
        elif strategy == 'mean':
            return self.data.mean(dim=-1)
        
        elif strategy == 'max':
            return self.data.max(dim=-1)[0]
        
        elif strategy == 'best':
            # Take path with highest average quality across batch
            best_path_idx = self.quality_scores.mean(dim=0).argmax()
            return self.data[:, :, best_path_idx]
        
        else:
            raise ValueError(f"Unknown collapse strategy: {strategy}")
    
    def to(self, device):
        """Move PathBundle to device"""
        return PathBundle(
            self.data.to(device),
            quality_scores=self.quality_scores.to(device)
        )
    
    def detach(self):
        """Detach PathBundle from computation graph"""
        return PathBundle(
            self.data.detach(),
            quality_scores=self.quality_scores.detach()
        )
    
    def __repr__(self):
        return (
            f"PathBundle(shape={self.shape}, "
            f"num_paths={self.num_paths}, "
            f"device={self.device})"
        )


def apply_per_path(fn: Callable, bundle: PathBundle) -> PathBundle:
    """
    Apply a function independently to each path in a PathBundle.
    
    Args:
        fn: Function that takes a 2D tensor [batch, features] and returns same shape
        bundle: Input PathBundle
        
    Returns:
        New PathBundle with fn applied to each path
    """
    new_paths = []
    for p in range(bundle.num_paths):
        path_data = bundle.get_path(p)  # [batch, features]
        transformed = fn(path_data)
        new_paths.append(transformed.unsqueeze(-1))  # [batch, features, 1]
    
    new_data = torch.cat(new_paths, dim=-1)  # [batch, features, paths]
    return PathBundle(new_data, quality_scores=bundle.quality_scores)


def concat_bundles(bundles: List[PathBundle], dim: int = -1) -> PathBundle:
    """
    Concatenate multiple PathBundles along a dimension.
    
    Args:
        bundles: List of PathBundles to concatenate
        dim: Dimension to concatenate along (-1 for paths, 1 for features)
        
    Returns:
        Concatenated PathBundle
    """
    if not bundles:
        raise ValueError("Cannot concatenate empty list of bundles")
    
    if dim == -1 or dim == 2:
        # Concatenate along paths dimension
        data = torch.cat([b.data for b in bundles], dim=2)
        scores = torch.cat([b.quality_scores for b in bundles], dim=1)
        return PathBundle(data, quality_scores=scores)
    
    elif dim == 1:
        # Concatenate along features dimension
        data = torch.cat([b.data for b in bundles], dim=1)
        # Quality scores remain the same (same paths, more features)
        return PathBundle(data, quality_scores=bundles[0].quality_scores)
    
    else:
        raise ValueError(f"Can only concatenate along dim 1 (features) or -1 (paths), got {dim}")