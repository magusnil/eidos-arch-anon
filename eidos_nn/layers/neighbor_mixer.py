import torch
import torch.nn as nn
from eidos_nn.layers.eidos_transform import eidosTransform
from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm

class eidosNeighborMixer(nn.Module):
    """
    Local Neighbor Interaction Layer.
    
    Provides local context by gathering [Left, Self, Right] token triplets
    and fusing them through a geometric eidosTransform. This gives each
    token awareness of its immediate neighbors before measurement/collapse,
    similar to a 1D convolution with kernel_size=3 but using rotation-based
    projections instead of dot-product kernels.
    
    Supports both bidirectional (classification) and causal (generation) modes.
    """
    def __init__(self, d_model, causal=False, matcher_mode="geometric", matcher_collapse="truncate"):
        super().__init__()
        self.causal = causal
        # Input: [Left, Self, Right] (or [Left2, Left1, Self] if causal) -> 3 * d_model
        # Output: d_model
        # num_rotation_planes=2 provides interaction between neighbor channels
        self.fusion = eidosTransform(
            d_model * 3,
            d_model,
            num_rotation_planes=2,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )
        self.norm = ModularPhaseNorm(d_model, base=7)
        
    def forward(self, x):
        # x: [batch, seq, dim]
        batch, seq, dim = x.shape
        
        # Shift to get neighbors (Zero padding at ends)
        # Left neighbor of i is i-1. shift right (positive) puts i-1 at i.
        left = torch.roll(x, shifts=1, dims=1)
        left[:, 0, :] = 0 # Boundary condition
        
        if self.causal:
            # CAUSAL MODE: Cannot see Right (Future)
            # Instead of [Left, Self, Right], we use [Left2, Left1, Self]
            # So 'left' is Left1 (i-1).
            # We need Left2 (i-2).
            left2 = torch.roll(x, shifts=2, dims=1)
            left2[:, 0:2, :] = 0
            
            # Triplet: [Left2, Left1, Self]
            # This preserves the "3-term" structure but looks backward.
            triplet = torch.cat([left2, left, x], dim=-1)
        else:
            # BIDIRECTIONAL MODE (Classification)
            # Right neighbor of i is i+1. shift left (negative) puts i+1 at i.
            right = torch.roll(x, shifts=-1, dims=1)
            right[:, -1, :] = 0 # Boundary condition
            
            # Concatenate [Left, Self, Right] -> [batch, seq, 3*dim]
            triplet = torch.cat([left, x, right], dim=-1)
        
        # Geometric Fusion
        mixed = self.fusion(triplet)
        
        # Normalize
        return self.norm(mixed)
