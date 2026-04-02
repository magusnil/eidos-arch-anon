import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from eidos_nn.layers.eidos_transform import eidosTransform
from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm
from eidos_nn.layers.neighbor_mixer import eidosNeighborMixer 

class MobiusCollapseLayer(nn.Module):
    """
    Implements a Möbius-topology Collapse Layer with fixed-point iteration.
    Replaces ProbableCollapseLayer. The 'collapsed' state feeds back into the
    observer and path decisions, creating continuous feedback within the layer.
    """
    def __init__(self, d_model: int, num_paths: int = 9, dropout: float = 0.1,
                 gumbel_tau: float = 1.0, causal: bool = False, num_twists: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_paths = num_paths
        self.gumbel_tau = gumbel_tau
        self.causal = causal
        self.num_twists = num_twists # Number of fixed-point iterations

        # Neighbor Mixer (Field Physics) - remains the same
        self.mixer = eidosNeighborMixer(d_model, causal=causal)

        # Observer now takes concatenated [current_form, current_collapsed_estimate]
        # This is where the Möbius twist happens: routing depends on consequences
        # d_model * 2 because of torch.cat([x, collapsed], dim=-1)
        d_mid_obs = d_model // 4
        self.observer_mlp = nn.Sequential(
            eidosTransform(d_model * 2, d_mid_obs, num_rotation_planes=1),
            nn.Tanh(), # Or other eidos-compatible activation
            eidosTransform(d_mid_obs, num_paths, num_rotation_planes=1)
        )

        # Paths (generate 9 potential transformations) - operate on twisted input
        self.paths = nn.ModuleList([
            nn.Sequential(
                eidosTransform(d_model, d_model // 4, num_rotation_planes=1), # Bottleneck
                eidosTransform(d_model // 4, d_model, num_rotation_planes=1) # Expand
            ) for _ in range(num_paths)
        ])
        
        # Form-to-form continuity (the 'twist back' operator)
        # This connects the previous 'collapsed' estimate to the paths' input
        self.form_twist = eidosTransform(d_model, d_model, num_rotation_planes=1)

        self.norm = ModularPhaseNorm(d_model, base=7)
        self.dropout = nn.Dropout(dropout)

        # Diagnostic: Track path selections (from final collapse)
        self.register_buffer('selection_counts', torch.zeros(num_paths))

    def get_selection_stats(self):
        """Return normalized selection percentages and reset counters."""
        total = self.selection_counts.sum()
        if total == 0:
            return torch.zeros_like(self.selection_counts)
        stats = self.selection_counts / total
        self.selection_counts.zero_()
        return stats

    def forward(self, x: torch.Tensor, initial_collapsed: Optional[torch.Tensor] = None):
        batch, seq, dim = x.shape

        # 0. Mix Neighbors (Context Awareness)
        x_mixed = self.mixer(x)

        # Initialize collapsed estimate for fixed-point iteration
        # If no previous collapsed state, start with zeros
        if initial_collapsed is None:
            # Need to ensure this tensor can be backproped through.
            # Start with a learnable state, or a direct transformation of x_mixed.
            # Starting with zeros_like(x_mixed) should be fine, it's a fixed initial point.
            collapsed_estimate = torch.zeros_like(x_mixed)
        else:
            # For generation, carry forward geometric memory
            collapsed_estimate = initial_collapsed

        # Fixed-point iteration (the Möbius twist)
        for t in range(self.num_twists):
            # Observer sees BOTH form (x_mixed) AND current collapsed estimate
            combined_input = torch.cat([x_mixed, collapsed_estimate], dim=-1)
            
            # Get routing logits from observer
            routing_logits = self.observer_mlp(combined_input)
            
            # Apply Gumbel-Softmax (soft selection for gradient flow)
            routing_weights = F.gumbel_softmax(routing_logits, tau=self.gumbel_tau, hard=False, dim=-1)

            # --- Compute Path Outputs ---
            # Paths operate on a 'twisted' input: current form + previous collapsed estimate
            twisted_paths_input = x_mixed + self.form_twist(collapsed_estimate)
            
            path_outputs = torch.stack([p(twisted_paths_input) for p in self.paths], dim=-1)
            
            # Collapse to new estimate
            new_collapsed_estimate = (path_outputs * routing_weights.unsqueeze(-2)).sum(dim=-1)
            
            # Normalize internal state to prevent 10^3 explosion per layer
            new_collapsed_estimate = self.norm(new_collapsed_estimate)
            
            # Update for next iteration
            collapsed_estimate = new_collapsed_estimate

        # Final collapse selection for diagnostics and final output
        # For the final output, we might want a hard selection for clarity, or keep it soft.
        # Let's keep it soft for now as per Gumbel-Softmax(hard=False)
        final_routing_logits = self.observer_mlp(torch.cat([x_mixed, collapsed_estimate], dim=-1))
        final_routing_weights_hard = F.gumbel_softmax(final_routing_logits, tau=self.gumbel_tau, hard=True, dim=-1)

        # Diagnostic: Track path selections from FINAL hard collapse
        with torch.no_grad():
            self.selection_counts += final_routing_weights_hard.sum(dim=[0, 1])

        # Final collapsed output
        final_collapsed_output = (path_outputs * final_routing_weights_hard.unsqueeze(-2)).sum(dim=-1)
        
        # Residual connection + Normalization
        out = self.norm(x + self.dropout(final_collapsed_output))
        
        # Normalize the collapsed estimate to prevent vertical explosion across layers
        # (The "Trillion Loss" Fix)
        collapsed_estimate = self.norm(collapsed_estimate)
        
        # Return the final collapsed output and the last collapsed_estimate for next step
        return out, collapsed_estimate