"""
Set-Valued Convolutional Neural Network
For proper image classification (CIFAR-10, etc.)
"""
import torch
import torch.nn as nn
from typing import Tuple


class SetValuedConv2d(nn.Module):
    """
    Set-valued convolutional layer: {-W, 0, +W}
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Single base weight (sign-symmetric)
        self.W_base = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.b_base = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, height, width]
        Returns:
            possibilities: [batch, out_channels, h', w', 3] where dim=-1 is {neg, zero, pos}
        """
        # Compute three branches
        out_neg = torch.nn.functional.conv2d(x, -self.W_base, -self.b_base,
                                             stride=self.stride, padding=self.padding)
        out_zero = torch.zeros_like(out_neg)
        out_pos = torch.nn.functional.conv2d(x, self.W_base, self.b_base,
                                            stride=self.stride, padding=self.padding)
        
        # Stack: [batch, out_channels, h', w', 3]
        possibilities = torch.stack([out_neg, out_zero, out_pos], dim=-1)
        
        return possibilities
    
    def _thread_select(self, possibilities: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        """
        Select from possibilities using pattern.
        
        Args:
            possibilities: [batch, channels, h, w, 3]
            pattern: [batch, channels, h, w] in {-1, 0, +1}
        Returns:
            selected: [batch, channels, h, w]
        """
        # Map pattern to indices: -1→0, 0→1, +1→2
        indices = (pattern + 1).long().unsqueeze(-1)  # [batch, channels, h, w, 1]
        
        # Gather
        selected = possibilities.gather(dim=-1, index=indices).squeeze(-1)
        
        return selected


class MaxAbsThreadingConv:
    """Threading for convolutional layers: select max abs value"""
    
    def __call__(self, possibilities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            possibilities: [batch, channels, h, w, 3]
        Returns:
            pattern: [batch, channels, h, w] in {-1, 0, +1}
        """
        abs_vals = possibilities.abs()
        max_idx = abs_vals.argmax(dim=-1)  # [batch, channels, h, w]
        pattern = max_idx - 1  # Convert 0,1,2 → -1,0,+1
        return pattern


class SetValuedCNN(nn.Module):
    """
    Set-valued CNN for image classification
    Architecture: Conv → Pool → Conv → Pool → FC → FC → Output
    """
    
    def __init__(self, num_classes=10, threading_strategy=None):
        super().__init__()
        self.threading_strategy = threading_strategy or MaxAbsThreadingConv()
        
        # Convolutional layers
        self.conv1 = SetValuedConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = SetValuedConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = SetValuedConv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (standard)
        # After 3 pools: 32x32 → 16x16 → 8x8 → 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, 32, 32]
        Returns:
            output: [batch, num_classes]
        """
        # Conv1 → Thread → Pool
        poss1 = self.conv1(x)  # [batch, 64, 32, 32, 3]
        pattern1 = self.threading_strategy(poss1)
        x = self.conv1._thread_select(poss1, pattern1)
        x = self.relu(x)
        x = self.pool(x)  # [batch, 64, 16, 16]
        
        # Conv2 → Thread → Pool
        poss2 = self.conv2(x)  # [batch, 128, 16, 16, 3]
        pattern2 = self.threading_strategy(poss2)
        x = self.conv2._thread_select(poss2, pattern2)
        x = self.relu(x)
        x = self.pool(x)  # [batch, 128, 8, 8]
        
        # Conv3 → Thread → Pool
        poss3 = self.conv3(x)  # [batch, 256, 8, 8, 3]
        pattern3 = self.threading_strategy(poss3)
        x = self.conv3._thread_select(poss3, pattern3)
        x = self.relu(x)
        x = self.pool(x)  # [batch, 256, 4, 4]
        
        # Flatten and FC
        x = x.view(x.size(0), -1)  # [batch, 256*4*4]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PathQualityCNN(nn.Module):
    """
    Path-Quality CNN: Maintains multiple paths through conv layers
    Uses quality scoring to prune paths
    """
    
    def __init__(self, num_classes=10, max_paths_per_layer=50):
        super().__init__()
        self.max_paths = max_paths_per_layer
        
        # Convolutional layers
        self.conv1 = SetValuedConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = SetValuedConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = SetValuedConv2d(128, 256, kernel_size=3, padding=1)
        
        # Path quality scorers (per conv layer)
        # Score spatial features to determine path quality
        self.scorer1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.scorer2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.scorer3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with path quality scoring
        
        Args:
            x: [batch, 3, 32, 32]
        Returns:
            output: [batch, num_classes]
        """
        batch_size = x.size(0)
        
        # Conv1: Generate paths, score, select best
        poss1 = self.conv1(x)  # [batch, 64, 32, 32, 3]
        x = self._quality_select(poss1, self.scorer1, self.conv1)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv2
        poss2 = self.conv2(x)  # [batch, 128, 16, 16, 3]
        x = self._quality_select(poss2, self.scorer2, self.conv2)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv3
        poss3 = self.conv3(x)  # [batch, 256, 8, 8, 3]
        x = self._quality_select(poss3, self.scorer3, self.conv3)
        x = self.relu(x)
        x = self.pool(x)
        
        # FC layers
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _quality_select(self, possibilities: torch.Tensor, scorer: nn.Module, 
                       conv_layer: SetValuedConv2d) -> torch.Tensor:
        """
        Select path using quality scores
        
        Args:
            possibilities: [batch, channels, h, w, 3]
            scorer: Quality scoring network
            conv_layer: Convolutional layer for thread_select
        Returns:
            selected: [batch, channels, h, w]
        """
        batch, channels, h, w, num_branches = possibilities.shape
        
        # Score each branch
        scores = []
        for branch_idx in range(num_branches):
            branch_data = possibilities[:, :, :, :, branch_idx]  # [batch, channels, h, w]
            score = scorer(branch_data)  # [batch, 1]
            scores.append(score)
        
        scores = torch.stack(scores, dim=-1)  # [batch, 1, 3]
        
        # Select branch with highest score
        best_branch = scores.argmax(dim=-1)  # [batch, 1]
        
        # Create pattern: broadcast best_branch to spatial dimensions
        pattern = best_branch.view(batch, 1, 1, 1).expand(batch, channels, h, w) - 1  # {-1, 0, +1}
        
        # Select using pattern
        selected = conv_layer._thread_select(possibilities, pattern)
        
        return selected


class SpatialPathQualityCNN(nn.Module):
    """
    Path-Quality CNN with PER-PIXEL path scoring
    FIX: Uses 1x1 convolutions to score paths spatially, not globally
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = SetValuedConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = SetValuedConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = SetValuedConv2d(128, 256, kernel_size=3, padding=1)
        
        # Per-PIXEL path quality scorers (1x1 convs)
        # Input: all 3 branches concatenated [channels*3]
        # Output: 3 scores per pixel [3, h, w]
        self.scorer1 = nn.Sequential(
            nn.Conv2d(64 * 3, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
        self.scorer2 = nn.Sequential(
            nn.Conv2d(128 * 3, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
        self.scorer3 = nn.Sequential(
            nn.Conv2d(256 * 3, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
        # Pooling
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
        batch_size = x.size(0)
        
        # Conv1: Per-pixel path selection
        poss1 = self.conv1(x)  # [batch, 64, 32, 32, 3]
        x = self._spatial_quality_select(poss1, self.scorer1, self.conv1)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv2
        poss2 = self.conv2(x)
        x = self._spatial_quality_select(poss2, self.scorer2, self.conv2)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv3
        poss3 = self.conv3(x)
        x = self._spatial_quality_select(poss3, self.scorer3, self.conv3)
        x = self.relu(x)
        x = self.pool(x)
        
        # FC layers
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Scale output
        x = x * self.output_scale
        
        return x
    
    def _spatial_quality_select(self, possibilities: torch.Tensor, scorer: nn.Module, 
                                conv_layer: SetValuedConv2d) -> torch.Tensor:
        """
        Select path using PER-PIXEL quality scores (FIX for global bug)
        
        Args:
            possibilities: [batch, channels, h, w, 3]
            scorer: 1x1 Conv that scores paths per pixel
            conv_layer: For thread_select
        Returns:
            selected: [batch, channels, h, w]
        """
        batch, channels, h, w, num_branches = possibilities.shape
        
        # Concatenate all branches for scoring
        # [batch, channels, h, w, 3] → [batch, channels*3, h, w]
        branches_cat = possibilities.permute(0, 1, 4, 2, 3)  # [batch, channels, 3, h, w]
        branches_cat = branches_cat.reshape(batch, channels * num_branches, h, w)
        
        # Score per pixel: [batch, channels*3, h, w] → [batch, 3, h, w]
        scores = scorer(branches_cat)  # [batch, 3, h, w] - DIFFERENT per pixel!
        
        # Select best branch PER PIXEL
        best_branch = scores.argmax(dim=1)  # [batch, h, w] - per-pixel choice
        
        # Expand to all channels: [batch, h, w] → [batch, channels, h, w]
        pattern = best_branch.unsqueeze(1).expand(batch, channels, h, w) - 1  # {-1, 0, +1}
        
        # Select using per-pixel pattern
        selected = conv_layer._thread_select(possibilities, pattern)
        
        return selected


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

