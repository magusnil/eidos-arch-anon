"""
TRUE Path-Preserving CNN
Maintains multiple paths across layers like MNIST PathBundle
NOT layer-local collapse!
"""
import torch
import torch.nn as nn
from set_valued_cnn import SetValuedConv2d


class PathScorer(nn.Module):
    """Score a spatial feature map for path quality"""
    
    def __init__(self, channels, height, width):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [batch, C, H, W] → [batch, C, 1, 1]
            nn.Flatten(),  # [batch, C]
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # [batch, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            score: [batch, 1]
        """
        return self.scorer(x)


class TruePathPreservingCNN(nn.Module):
    """
    Path-Preserving CNN that ACTUALLY maintains paths across layers.
    Like MNIST PathBundle but for spatial data.
    
    Key difference from SpatialPathQualityCNN:
    - This keeps MULTIPLE paths through entire network
    - SpatialPQ collapsed to 1 path at each layer (still layer-local!)
    """
    
    def __init__(self, num_classes=10, max_paths_per_layer=(3, 9, 27)):
        super().__init__()
        self.max_paths_per_layer = max_paths_per_layer
        
        # Conv layers
        self.conv1 = SetValuedConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = SetValuedConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = SetValuedConv2d(128, 256, kernel_size=3, padding=1)
        
        # Path scorers (evaluate spatial features)
        self.scorer1 = PathScorer(64, 32, 32)
        self.scorer2 = PathScorer(128, 16, 16)
        self.scorer3 = PathScorer(256, 8, 8)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Output scaling (learned)
        self.output_scale = nn.Parameter(torch.ones(1) * 10.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, 32, 32]
        Returns:
            output: [batch, num_classes]
        """
        batch = x.size(0)
        
        # Start with 1 path: [batch, 3, 32, 32] → [batch, 3, 32, 32, 1]
        x = x.unsqueeze(-1)
        
        # Conv1: 1 path → 3 paths
        x = self._expand_conv_paths(x, self.conv1)  # [batch, 64, 32, 32, 3]
        x = self._relu_paths(x)
        x = self._pool_paths(x)  # [batch, 64, 16, 16, 3]
        
        # Score and prune to max_paths_per_layer[0]
        x = self._score_and_prune_paths(x, self.scorer1, keep=self.max_paths_per_layer[0])
        
        # Conv2: 3 paths → 9 paths
        x = self._expand_conv_paths(x, self.conv2)  # [batch, 128, 16, 16, 9]
        x = self._relu_paths(x)
        x = self._pool_paths(x)  # [batch, 128, 8, 8, 9]
        
        # Score and prune
        x = self._score_and_prune_paths(x, self.scorer2, keep=self.max_paths_per_layer[1])
        
        # Conv3: 9 paths → 27 paths
        x = self._expand_conv_paths(x, self.conv3)  # [batch, 256, 8, 8, 27]
        x = self._relu_paths(x)
        x = self._pool_paths(x)  # [batch, 256, 4, 4, 27]
        
        # Score and prune
        x = self._score_and_prune_paths(x, self.scorer3, keep=self.max_paths_per_layer[2])
        
        # Flatten each path and process through FC
        batch, C, H, W, num_paths = x.shape
        x = x.permute(0, 4, 1, 2, 3)  # [batch, num_paths, C, H, W]
        x = x.reshape(batch * num_paths, C * H * W)  # Treat paths as batch
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [batch*num_paths, num_classes]
        
        # Reshape back to paths
        x = x.view(batch, num_paths, -1)  # [batch, num_paths, num_classes]
        
        # Average over paths (ensemble)
        x = x.mean(dim=1)  # [batch, num_classes]
        
        # Scale output
        x = x * self.output_scale
        
        return x
    
    def _expand_conv_paths(self, x: torch.Tensor, conv_layer: SetValuedConv2d) -> torch.Tensor:
        """
        Expand paths through a conv layer.
        Each input path generates 3 output paths.
        
        Args:
            x: [batch, C_in, H, W, num_paths_in]
            conv_layer: SetValuedConv2d
        Returns:
            expanded: [batch, C_out, H', W', num_paths_in * 3]
        """
        batch, C_in, H, W, num_paths_in = x.shape
        
        all_branches = []
        for path_idx in range(num_paths_in):
            # Extract one path
            path = x[:, :, :, :, path_idx]  # [batch, C_in, H, W]
            
            # Apply conv to get 3 branches
            branches = conv_layer(path)  # [batch, C_out, H', W', 3]
            
            all_branches.append(branches)
        
        # Concatenate all branches: [batch, C_out, H', W', num_paths_in * 3]
        expanded = torch.cat(all_branches, dim=-1)
        
        return expanded
    
    def _relu_paths(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU to all paths"""
        return torch.relu(x)
    
    def _pool_paths(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply max pooling to each path independently.
        
        Args:
            x: [batch, C, H, W, num_paths]
        Returns:
            pooled: [batch, C, H/2, W/2, num_paths]
        """
        batch, C, H, W, num_paths = x.shape
        
        # Reshape to treat paths as batch: [batch*num_paths, C, H, W]
        x_flat = x.permute(0, 4, 1, 2, 3).reshape(batch * num_paths, C, H, W)
        
        # Pool
        pooled_flat = self.pool(x_flat)  # [batch*num_paths, C, H/2, W/2]
        
        # Reshape back: [batch, num_paths, C, H/2, W/2] → [batch, C, H/2, W/2, num_paths]
        _, C, H_new, W_new = pooled_flat.shape
        pooled = pooled_flat.view(batch, num_paths, C, H_new, W_new)
        pooled = pooled.permute(0, 2, 3, 4, 1)  # [batch, C, H/2, W/2, num_paths]
        
        return pooled
    
    def _score_and_prune_paths(self, x: torch.Tensor, scorer: PathScorer, keep: int) -> torch.Tensor:
        """
        Score each path and keep top-k.
        THIS IS WHERE GRADIENTS FLOW TO SCORER!
        
        Args:
            x: [batch, C, H, W, num_paths]
            scorer: PathScorer network
            keep: Number of paths to keep
        Returns:
            pruned: [batch, C, H, W, keep]
        """
        batch, C, H, W, num_paths = x.shape
        
        if num_paths <= keep:
            return x
        
        # Score each path
        scores = []
        for path_idx in range(num_paths):
            path = x[:, :, :, :, path_idx]  # [batch, C, H, W]
            score = scorer(path)  # [batch, 1]
            scores.append(score)
        
        scores = torch.cat(scores, dim=1)  # [batch, num_paths]
        
        # Select top-k paths by quality
        _, top_indices = scores.topk(keep, dim=1)  # [batch, keep]
        
        # Gather top paths
        # Expand indices: [batch, keep] → [batch, C, H, W, keep]
        top_indices = top_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [batch, 1, 1, 1, keep]
        top_indices = top_indices.expand(batch, C, H, W, keep)
        
        pruned = x.gather(dim=-1, index=top_indices)
        
        return pruned


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

