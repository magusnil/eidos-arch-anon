"""
Hybrid Set-Valued CNN
Layer-Local threading for Conv layers (spatial per-pixel decisions)
Path-Quality for FC layers (global path decisions)
"""
import torch
import torch.nn as nn
from set_valued_cnn import SetValuedConv2d, MaxAbsThreadingConv


class PathBundle:
    """Bundle of paths for FC layers (3D: [batch, features, num_paths])"""
    
    def __init__(self, data: torch.Tensor):
        self.data = data  # [batch, features, num_paths]
    
    @property
    def num_paths(self):
        return self.data.shape[-1]
    
    def apply_activation(self, fn):
        return PathBundle(fn(self.data))


class PathPreservingFC(nn.Module):
    """FC layer that expands paths: num_paths_in -> num_paths_in * 3"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Single base weight (sign-symmetric)
        self.W_base = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.b_base = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, path_bundle: PathBundle) -> PathBundle:
        """
        Args:
            path_bundle: [batch, in_features, num_paths_in]
        Returns:
            PathBundle: [batch, out_features, num_paths_in * 3]
        """
        batch, in_feat, num_paths = path_bundle.data.shape
        
        # Transpose for matmul: [batch, num_paths, in_features]
        paths_t = path_bundle.data.transpose(1, 2)
        
        # Compute three branches for each path
        out_neg = paths_t @ (-self.W_base) + (-self.b_base)  # [batch, num_paths, out_features]
        out_zero = torch.zeros_like(out_neg)
        out_pos = paths_t @ self.W_base + self.b_base
        
        # Stack branches: [batch, num_paths, out_features, 3]
        branches = torch.stack([out_neg, out_zero, out_pos], dim=-1)
        
        # Reshape to [batch, out_features, num_paths * 3]
        all_paths = branches.transpose(1, 2).reshape(batch, self.out_features, -1)
        
        return PathBundle(all_paths)


class HybridSetValuedCNN(nn.Module):
    """
    Hybrid architecture:
    - Conv layers: Layer-Local threading (MaxAbs per-pixel)
    - FC layers: Path-Quality scoring and pruning
    """
    
    def __init__(self, num_classes=10, max_paths=50):
        super().__init__()
        self.max_paths = max_paths
        self.threading_conv = MaxAbsThreadingConv()
        
        # Convolutional layers (Layer-Local)
        self.conv1 = SetValuedConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = SetValuedConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = SetValuedConv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # FC layers (Path-Quality)
        self.fc1 = PathPreservingFC(256 * 4 * 4, 512)
        self.fc2 = PathPreservingFC(512, num_classes)
        
        # Path quality scorers for FC layers
        self.scorer1 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.scorer2 = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, 32, 32]
        Returns:
            output: [batch, num_classes]
        """
        batch_size = x.size(0)
        
        # ========== CONVOLUTIONAL LAYERS (Layer-Local) ==========
        # Conv1 + MaxAbs threading (per-pixel)
        poss1 = self.conv1(x)
        pattern1 = self.threading_conv(poss1)
        x = self.conv1._thread_select(poss1, pattern1)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv2
        poss2 = self.conv2(x)
        pattern2 = self.threading_conv(poss2)
        x = self.conv2._thread_select(poss2, pattern2)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv3
        poss3 = self.conv3(x)
        pattern3 = self.threading_conv(poss3)
        x = self.conv3._thread_select(poss3, pattern3)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten: [batch, 256*4*4]
        x = x.view(batch_size, -1)
        
        # ========== FC LAYERS (Path-Quality) ==========
        # Start with single path
        paths = PathBundle(x.unsqueeze(-1))  # [batch, 256*4*4, 1]
        
        # FC1: Expand to 3 paths
        paths = self.fc1(paths)  # [batch, 512, 3]
        
        # Score and prune
        if paths.num_paths > self.max_paths:
            paths = self._quality_prune(paths, self.scorer1, keep=self.max_paths)
        
        # Activation
        paths = paths.apply_activation(torch.tanh)
        paths = PathBundle(torch.dropout(paths.data, p=0.5, train=self.training))
        
        # FC2: Expand paths
        paths = self.fc2(paths)  # [batch, num_classes, num_paths]
        
        # Final: Quality-weighted collapse
        output = self._quality_weighted_output(paths, self.scorer2)
        
        return output
    
    def _quality_prune(self, paths: PathBundle, scorer: nn.Module, keep: int) -> PathBundle:
        """
        Prune paths based on quality scores
        
        Args:
            paths: [batch, features, num_paths]
            scorer: Quality scoring network
            keep: Number of paths to keep
        Returns:
            pruned: [batch, features, keep]
        """
        batch, features, num_paths = paths.data.shape
        
        if num_paths <= keep:
            return paths
        
        # Score each path: [batch, num_paths]
        scores = []
        for path_idx in range(num_paths):
            path_data = paths.data[:, :, path_idx]  # [batch, features]
            score = scorer(path_data)  # [batch, 1]
            scores.append(score)
        
        scores = torch.stack(scores, dim=-1).squeeze(1)  # [batch, num_paths]
        
        # Select top-k paths per batch element
        top_indices = scores.topk(keep, dim=-1).indices  # [batch, keep]
        
        # Gather paths
        # Expand indices: [batch, keep] -> [batch, features, keep]
        expanded_indices = top_indices.unsqueeze(1).expand(batch, features, keep)
        pruned_data = paths.data.gather(dim=2, index=expanded_indices)
        
        return PathBundle(pruned_data)
    
    def _quality_weighted_output(self, paths: PathBundle, scorer: nn.Module) -> torch.Tensor:
        """
        Weighted average of paths based on quality scores
        
        Args:
            paths: [batch, features, num_paths]
            scorer: Quality scoring network
        Returns:
            output: [batch, features]
        """
        batch, features, num_paths = paths.data.shape
        
        # Score each path
        scores = []
        for path_idx in range(num_paths):
            path_data = paths.data[:, :, path_idx]
            score = scorer(path_data)
            scores.append(score)
        
        scores = torch.stack(scores, dim=-1).squeeze(1)  # [batch, num_paths]
        weights = torch.softmax(scores, dim=-1)  # [batch, num_paths]
        
        # Weighted sum: [batch, features, num_paths] * [batch, 1, num_paths] -> [batch, features]
        output = (paths.data * weights.unsqueeze(1)).sum(dim=-1)
        
        return output


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

