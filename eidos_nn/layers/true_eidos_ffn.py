"""
EIDOS FEED-FORWARD NETWORK (Set-Valued, No nn.Linear)

Feed-Forward Network for sequence data using geometric eidosTransform
projections instead of nn.Linear.

Architecture:
  - SEPARATE eidosTransform per path (independent parameterization).
  - ModularPhaseNorm between projection stages.
  - Hierarchical quality scoring for path collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
from pathlib import Path

# Add parent to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .hierarchical_scorer import HierarchicalPathScorer
from eidos_nn.layers.eidos_transform import eidosTransform, eidosSequential

# Direct import for ModularPhaseNorm
import importlib.util
spec = importlib.util.spec_from_file_location(
    "modular_phase_norm",
    PROJECT_ROOT / "utils" / "modular_phase_norm.py"
)
phase_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase_module)
ModularPhaseNorm = phase_module.ModularPhaseNorm


class TrueeidosFFN(nn.Module):
    """
    True eidos Feed-Forward Network with separate transformations per path.

    Follows the same pattern as TrueeidosConv2D:
      - SEPARATE eidos transformations for each path (not shared!)
      - Path-specific biases
      - Context state (optional)
      - Hierarchical quality scorer

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension
        num_paths: Number of parallel paths (default: 9)
        dropout: Dropout probability
        use_context: Enable context-dependent state
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_paths: int = 9,
        dropout: float = 0.0,
        use_context: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_paths = num_paths
        self.use_context = use_context

        # ================================================================
        # MULTIPLE eidos FFNs - One per path (NOT shared!)
        # ================================================================
        # Replaced nn.Linear with eidosTransform + ModularPhaseNorm
        # This preserves geometric structure instead of flattening
        self.path_ffns = nn.ModuleList([
            eidosSequential(
                eidosTransform(d_model, d_ff, num_rotation_planes=4),
                ModularPhaseNorm(d_ff, base=7),
                eidosTransform(d_ff, d_model, num_rotation_planes=4),
                ModularPhaseNorm(d_model, base=7) # Normalization after projection back
            )
            for _ in range(num_paths)
        ])

        # Path-specific biases (context-dependent)
        self.path_biases = nn.Parameter(torch.zeros(num_paths, d_model))

        # ================================================================
        # CONTEXT STATE (σ) - For eidos operations
        # ================================================================
        if use_context:
            # Learnable context state for each path
            self.context_state = nn.Parameter(
                torch.randn(num_paths, d_model) * 0.01
            )
        else:
            self.register_buffer('context_state', torch.zeros(num_paths, d_model))

        # ================================================================
        # HIERARCHICAL PATH SCORER - Quality-weighted collapse
        # ================================================================
        # This now uses Eidos internally as well
        team_size = 3 if num_paths == 9 else 9
        self.quality_scorer = HierarchicalPathScorer(
            num_paths=num_paths,
            feature_dim=d_model,
            team_size=team_size,
            use_overhead=True
        )

        # Dropout (Geometric dropout would be better, but kept at 0.0 as requested)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Set-valued feed-forward with separate paths.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # ================================================================
        # STEP 1: Generate set of possible transformations (one per path)
        # ================================================================
        path_features = []

        for path_idx, path_ffn in enumerate(self.path_ffns):
            # Apply path-specific eidos transform
            feature = path_ffn(x)  # [batch, seq_len, d_model]

            # Add path-specific bias
            feature = feature + self.path_biases[path_idx].view(1, 1, -1)

            # Apply context (simplified R3/R4 operations)
            if self.use_context:
                context = self.context_state[path_idx].view(1, 1, -1)
                # Non-linear interaction with context
                feature = feature + 0.1 * context * torch.tanh(feature)

            # Dropout
            feature = self.dropout(feature)

            path_features.append(feature)

        # Stack all paths: [batch, seq_len, d_model, num_paths]
        path_tensor = torch.stack(path_features, dim=-1)

        # ================================================================
        # STEP 2: Hierarchical collapse via quality scoring
        # ================================================================
        # Reshape for quality scoring: [batch*seq_len, d_model, num_paths]
        batch_seq = batch * seq_len
        features_flat = path_tensor.reshape(batch_seq, d_model, self.num_paths)

        # Hierarchical scoring and collapse
        output_flat = self.quality_scorer(features_flat)  # [batch*seq_len, d_model]

        # Reshape back to [batch, seq_len, d_model]
        output = output_flat.reshape(batch, seq_len, d_model)

        return output


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRUE eidos FFN - Following Working Conv Pattern")
    print("=" * 70)
    print("\nKey difference from HierarchicalSetValuedFFN:")
    print("  ✓ Separate linear layers for EACH path (like TrueeidosConv2D)")
    print("  ✗ NOT reusing same fc2 layer 3 times!")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Test with AIMO dimensions
    batch = 4
    seq_len = 512
    d_model = 512
    d_ff = 2048

    print(f"Dimensions (matching AIMO):")
    print(f"  Batch: {batch}")
    print(f"  Seq len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}\n")

    print("Creating TrueeidosFFN...")
    ffn = TrueeidosFFN(
        d_model=d_model,
        d_ff=d_ff,
        num_paths=9,
        dropout=0.1
    ).to(device)
    print("✓ FFN created\n")

    print("Creating test input...")
    x = torch.randn(batch, seq_len, d_model, device=device)
    print(f"Input shape: {x.shape}\n")

    print("Running forward pass...")
    output = ffn(x)
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass complete!\n")

    print("Testing backward pass...")
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass complete!\n")

    # Count parameters
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"Total parameters: {total_params:,}\n")

    print("=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThis FFN follows the working TrueeidosConv2D pattern:")
    print("  - Separate transformations per path (ModuleList)")
    print("  - Path-specific biases and context")
    print("  - Hierarchical quality scorer")
    print("  - Should NOT hang like HierarchicalSetValuedFFN!")
    print()

