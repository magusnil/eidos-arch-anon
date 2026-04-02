import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Handle imports assuming we are running from root or within module
try:
    from .eidos_transform import eidosTransform
    from ..utils.modular_phase_norm import ModularPhaseNorm
      # experimental to not use, as is not useful unless the dataset calls for it (e.g., CIFAR10)
    
    from ..layers.neighbor_mixer import eidosNeighborMixer
except ImportError:
    # Fallback for direct execution or different path structure
    from eidos_nn.layers.eidos_transform import eidosTransform
    from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm
     # experimental to not use, as is not useful
    
    
    from eidos_nn.layers.neighbor_mixer import eidosNeighborMixer

class ProbableCollapseLayer(nn.Module):
    """
    Implements the Measurement-Driven Collapse (The Observer Effect).
    
    Architecture:
    1. Candidate Generation: Produces 'num_paths' parallel transformations.
    2. Observation: An Observer module scores each candidate path.
    3. Collapse: Hard Gumbel-Softmax selection of the highest-scored path.
    
    Efficiency:
    Uses a bottleneck architecture for paths to generate variation without massive parameter explosion.
    """
    def __init__(
        self,
        d_model,
        num_paths=9,
        dropout=0.1,
        gumbel_tau=1.0,
        causal=False,
        matcher_mode="geometric",
        matcher_collapse="truncate",
    ):
        super().__init__()
        self.num_paths = num_paths
        self.gumbel_tau = gumbel_tau
        
        # Neighbor Mixer (Field Physics)
        # Provides context awareness before observation/collapse
        self.mixer = eidosNeighborMixer(
            d_model,
            causal=causal,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )
        
        # 1. Candidate Path Generator
        # Generates 9 distinct "potential realities" from the input.
        # Uses bottleneck (d_model -> d_mid -> d_model) to keep params low.
        d_mid = d_model // 4 
        self.paths = nn.ModuleList([
            nn.Sequential(
                eidosTransform(
                    d_model,
                    d_mid,
                    num_rotation_planes=1,
                    matcher_mode=matcher_mode,
                    matcher_collapse=matcher_collapse,
                ), # Compress/Encode
                # nn.GELU(), # Removed: Classical smoothing breaks scale-invariance
                eidosTransform(
                    d_mid,
                    d_model,
                    num_rotation_planes=1,
                    matcher_mode=matcher_mode,
                    matcher_collapse=matcher_collapse,
                )  # Expand/Decode
            ) for _ in range(num_paths)
        ])
        
        # 2. The Observer (Measurement Apparatus)
        # Measures the input context to decide which path is "Real".
        # The observer is lightweight.
        self.observer = nn.Sequential(
            eidosTransform(
                d_model,
                d_mid,
                num_rotation_planes=1,
                matcher_mode=matcher_mode,
                matcher_collapse=matcher_collapse,
            ),
            nn.Tanh(), # Stability activation
            eidosTransform(
                d_mid,
                num_paths,
                num_rotation_planes=1,
                matcher_mode=matcher_mode,
                matcher_collapse=matcher_collapse,
            )
        )
        
        self.norm = ModularPhaseNorm(d_model, base=7)
        self.dropout = nn.Dropout(dropout)
        
        # ModularPhaseNorm for noise component (modular phase normalization for the noise component)
        self.noise_phase_norm = ModularPhaseNorm(num_paths, base=7)
        
        
        # Diagnostic: Track path selections
        self.register_buffer('selection_counts', torch.zeros(num_paths))

    def get_selection_stats(self):
        """Return normalized selection percentages and reset counters."""
        total = self.selection_counts.sum()
        if total == 0:
            return torch.zeros_like(self.selection_counts)
        stats = self.selection_counts / total
        self.selection_counts.zero_()
        return stats

    def forward(self, x):
        # x: [batch, seq, dim]
        batch, seq, dim = x.shape
        
        # 0. Mix Neighbors (Context Awareness)
        # The field interacts locally before being measured.
        x_mixed = self.mixer(x)
        
        # 1. Generate Candidate Paths
        # Compute all candidate path outputs in parallel.
        # [batch, seq, num_paths, dim]
        # Paths now act on the MIXED context
        path_outputs = [path(x_mixed) for path in self.paths] 
        stacked_paths = torch.stack(path_outputs, dim=2) 
        

        # 2. Observation (Measurement)
        # The observer scores the suitability of each path given the context x.
        # Observer looks at MIXED context
        # [batch, seq, num_paths]
        observer_scores = self.observer(x_mixed) 
        
        # 3. Collapse (Selection)
        if self.training:
            noisy_scores = observer_scores
            
            # Gumbel-Softmax allowing gradients to flow through the selection process.
            # Approximates a hard selection (one-hot).
            weights = F.gumbel_softmax(noisy_scores, tau=self.gumbel_tau, hard=True, dim=-1)
            
            # Diagnostic: Track soft/hard selections
            # Since we use hard=True, weights are one-hot. Sum over batch/seq
            with torch.no_grad():
                # Sum over batch(0) and seq(1)
                current_counts = weights.sum(dim=[0, 1])
                self.selection_counts += current_counts
                
        else:
            # Hard Collapse: Explicitly select the max score index.
            indices = torch.argmax(observer_scores, dim=-1) # [batch, seq]
            weights = F.one_hot(indices, num_classes=self.num_paths).float()
            
        # Expand weights for broadcasting: [batch, seq, num_paths, 1]
        weights = weights.unsqueeze(-1)
        
        # Actualize the selected reality
        # Since weights are effectively one-hot, this selects exactly one path's output per token.
        collapsed_reality = torch.sum(stacked_paths * weights, dim=2)
        
        # Residual connection (Stability) + Normalization
        # R3 (Carry) axiom: Previous state (x) carries forward, modified by the collapsed transformation.
        out = self.norm(x + self.dropout(collapsed_reality))
        
        return out

