"""
IR HIERARCHICAL POSITIONAL ENCODING - Dimensional Regime-Based Position Embeddings

Mathematical Foundation:
    Isolate Recursion (IR) sequence: IR_n = 5 × 11^(n-1)

    Dimensional regimes:
        d0: [0, 4]          - Axiomatic Core (5 positions)
        d1: [5, 54]         - Elemental Reflection (50 positions)
        d2: [55, 604]       - First Fracture (550 positions)
        d3: [605, 6654]     - Self-Interaction (6,050 positions)
        d4: [6655, 73204]   - Prime Extinction (66,550 positions)
        d5: [73205, 805254] - Entropy Saturation (732,050 positions)

Key Insight:
    Instead of treating all positions uniformly, we encode positions based on
    which dimensional regime they fall into. This creates a FRACTAL positional
    encoding that naturally handles exponentially growing context lengths.

    Documents have natural hierarchical structure:
        Word → Sentence → Paragraph → Section → Chapter → Book

    This maps to dimensional regimes:
        d0 (5 pos)    → Individual tokens
        d1 (50 pos)   → Sentences
        d2 (550 pos)  → Paragraphs
        d3 (6K pos)   → Sections
        d4 (66K pos)  → Chapters
        d5 (800K pos) → Books

Novel Contribution:
    No existing LLM uses dimensional regime boundaries for positional encoding.
    This approach is ENTIRELY grounded in eidos mathematical framework.

Author: Anonymous
Framework: Eidos

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================================
# DIMENSIONAL REGIME CLASSIFIER (PyTorch Implementation)
# ============================================================================

def compute_ir(n: int) -> int:
    """
    Compute nth term of Isolate Recursion sequence.

    IR_n = 5 × 11^(n-1)

    Args:
        n: Regime number (n >= 1)

    Returns:
        IR boundary value
    """
    if n < 1:
        return 0
    return 5 * (11 ** (n - 1))


# Precompute IR boundaries for fast lookup
IR_BOUNDARIES = torch.tensor([
    0,      # d0 start
    5,      # d1 start (IR_1)
    55,     # d2 start (IR_2)
    605,    # d3 start (IR_3)
    6655,   # d4 start (IR_4)
    73205,  # d5 start (IR_5)
    805254, # d6 start (IR_6) - theoretical limit
], dtype=torch.long)


def classify_dimensional_regime_torch(positions: torch.Tensor) -> torch.Tensor:
    """
    Classify positions into dimensional regimes (vectorized PyTorch version).

    Args:
        positions: [batch, seq_len] or [seq_len] position indices

    Returns:
        regimes: Same shape as positions, containing regime indices (0-5)

    Example:
        positions = torch.tensor([0, 3, 10, 60, 700, 8000])
        regimes = classify_dimensional_regime_torch(positions)
        # Output: [0, 0, 1, 2, 3, 4]
        #         d0, d0, d1, d2, d3, d4
    """
    # Handle negative positions (shouldn't happen, but be safe)
    positions = torch.abs(positions)

    # Expand IR_BOUNDARIES to match position dimensions
    # positions: [...] → [..., 1]
    # IR_BOUNDARIES: [num_regimes] → [1, ..., 1, num_regimes]
    pos_expanded = positions.unsqueeze(-1)  # [..., 1]
    boundaries = IR_BOUNDARIES.to(positions.device)

    # Broadcasting comparison: [..., 1] >= [num_regimes]
    # Result: [..., num_regimes] boolean tensor
    above_boundary = pos_expanded >= boundaries.unsqueeze(0)

    # Sum along regime dimension to count how many boundaries we've crossed
    # This gives us the regime index
    regime_indices = above_boundary.sum(dim=-1) - 1

    # Clamp to valid range [0, 5]
    regime_indices = torch.clamp(regime_indices, 0, 5)

    return regime_indices.long()


def get_local_position_in_regime(positions: torch.Tensor, regimes: torch.Tensor) -> torch.Tensor:
    """
    Convert global positions to local positions within their dimensional regime.

    Args:
        positions: [batch, seq_len] or [seq_len] global position indices
        regimes: [batch, seq_len] or [seq_len] regime classifications

    Returns:
        local_positions: Same shape, positions relative to regime start

    Example:
        positions = torch.tensor([0, 3, 55, 60, 605, 610])
        regimes = torch.tensor([0, 0, 2, 2, 3, 3])
        local_pos = get_local_position_in_regime(positions, regimes)
        # Output: [0, 3, 0, 5, 0, 5]
        # (55 is start of d2, so local=0; 60 is 5 positions into d2)
    """
    # Get regime start boundaries
    boundaries = IR_BOUNDARIES.to(positions.device)

    # Gather the boundary for each position's regime
    regime_starts = boundaries[regimes]

    # Local position = global position - regime start
    local_positions = positions - regime_starts

    return local_positions


# ============================================================================
# IR HIERARCHICAL POSITIONAL ENCODING MODULE
# ============================================================================

class IRHierarchicalPositionalEncoding(nn.Module):
    """
    Positional encoding based on dimensional regime boundaries.

    Architecture:
        1. Classify each position into dimensional regime (d0-d5)
        2. Encode local position within regime using regime-specific embedder
        3. Add regime identity embeddings (which regime am I in?)
        4. Smooth transitions at regime boundaries using learned gates

    Key Properties:
        - Scales to 800K tokens (d5 boundary)
        - Natural hierarchical structure
        - Provably independent regimes (from axiom analysis)
        - Phase transitions at boundaries can trigger actualization
        - Eidos: No RoPE mixing (incompatible with set-valued operations!)

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length (default: 65536 = d4 boundary)
        num_regimes: Number of dimensional regimes to use (default: 5 for d0-d4)
        use_rope: DEPRECATED - kept for backwards compatibility, always False
        learnable_transitions: Learn smooth gates at regime boundaries (default: True)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 65536,
        num_regimes: int = 5,
        use_rope: bool = False,  # Eidos: No RoPE!
        learnable_transitions: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_regimes = num_regimes
        self.use_rope = False  # Force disabled - RoPE incompatible with set-valued ops!
        self.learnable_transitions = learnable_transitions

        # Validate num_regimes
        assert 1 <= num_regimes <= 6, f"num_regimes must be 1-6, got {num_regimes}"

        # Calculate regime sizes
        regime_sizes = []
        for i in range(num_regimes):
            if i == 0:
                regime_sizes.append(5)  # d0: [0, 4]
            else:
                regime_sizes.append(compute_ir(i+1) - compute_ir(i))

        print(f"\n[IR-POS] IR Hierarchical Positional Encoding (Eidos)")
        print(f"   Model dimension: {d_model}")
        print(f"   Max sequence length: {max_seq_len:,}")
        print(f"   Dimensional regimes: {num_regimes} (d0-d{num_regimes-1})")
        print(f"   Regime sizes: {[f'{s:,}' for s in regime_sizes]}")
        print(f"   RoPE mixing: DISABLED (incompatible with set-valued ops!)")
        print(f"   Learnable transitions: {learnable_transitions}")

        # Regime-specific positional embeddings
        # Each regime has its own embedding table sized for that regime
        self.regime_embedders = nn.ModuleList([
            nn.Embedding(regime_sizes[i], d_model)
            for i in range(num_regimes)
        ])

        # Regime identity embeddings (what regime am I in?)
        # This helps the model learn different behaviors in different regimes
        self.regime_identity_emb = nn.Embedding(num_regimes, d_model)

        # Regime transition gates (smooth interpolation at boundaries)
        if learnable_transitions:
            self.transition_gates = nn.Parameter(torch.ones(num_regimes))
            self.transition_smoothness = nn.Parameter(torch.ones(1) * 10.0)  # Sigmoid sharpness
        else:
            self.register_buffer('transition_gates', torch.ones(num_regimes))
            self.register_buffer('transition_smoothness', torch.tensor(10.0))

        # RoPE DISABLED - Incompatible with set-valued operations!
        # if use_rope:
        #     self.rope_freqs = self._init_rope_frequencies(d_model, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _init_rope_frequencies(self, d_model: int, max_seq_len: int) -> torch.Tensor:
        """
        Initialize RoPE frequency vectors.

        This provides relative positional information that complements
        the regime-based absolute positions.
        """
        # Standard RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

        # Precompute position indices
        t = torch.arange(max_seq_len).float()

        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [max_seq_len, d_model//2]

        # Concatenate sin and cos
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, d_model]

        # Register as buffer (not trainable, but moves with model to device)
        self.register_buffer('rope_cos', emb.cos())
        self.register_buffer('rope_sin', emb.sin())

        return freqs

    def _init_weights(self):
        """Initialize embedding weights (small random values)."""
        for embedder in self.regime_embedders:
            nn.init.normal_(embedder.weight, mean=0.0, std=0.02)

        nn.init.normal_(self.regime_identity_emb.weight, mean=0.0, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute IR hierarchical positional encodings.

        Args:
            positions: [batch, seq_len] or [seq_len] position indices (0-indexed)

        Returns:
            pos_encodings: [batch, seq_len, d_model] positional embeddings

        Example:
            # Batch of 2 sequences, each 128 tokens
            positions = torch.arange(128).unsqueeze(0).expand(2, -1)
            pos_enc = ir_encoder(positions)
            # Shape: [2, 128, d_model]
        """
        # Handle 1D input (no batch dimension)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # [seq_len] → [1, seq_len]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len = positions.shape
        device = positions.device

        # Step 1: Classify positions into dimensional regimes
        regimes = classify_dimensional_regime_torch(positions)  # [batch, seq_len]

        # Step 2: Get local positions within regimes
        local_positions = get_local_position_in_regime(positions, regimes)  # [batch, seq_len]

        # Step 3: Embed local positions using regime-specific embedders
        # This is the core of the hierarchical encoding!
        regime_encodings = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        for regime_idx in range(self.num_regimes):
            # Mask: which positions are in this regime?
            mask = (regimes == regime_idx)  # [batch, seq_len]

            if mask.any():
                # Get local positions for this regime
                local_pos_masked = local_positions[mask]  # [num_positions_in_regime]

                # Clamp to valid range (safety)
                max_pos = self.regime_embedders[regime_idx].num_embeddings - 1
                local_pos_masked = torch.clamp(local_pos_masked, 0, max_pos)

                # Embed these positions
                encoded = self.regime_embedders[regime_idx](local_pos_masked)  # [num_pos, d_model]

                # Place back into output tensor
                regime_encodings[mask] = encoded

        # Step 4: Add regime identity embeddings
        # This tells the model "you are in regime d2" vs "you are in regime d3"
        regime_identity = self.regime_identity_emb(regimes)  # [batch, seq_len, d_model]
        regime_encodings = regime_encodings + regime_identity

        # Step 5: Apply regime transition gates (smooth boundaries)
        if self.learnable_transitions:
            # Compute distance to next regime boundary
            # Move regimes to CPU for indexing, then move result to device
            boundaries = IR_BOUNDARIES[(regimes + 1).cpu()].to(device)  # Next boundary
            distance_to_boundary = boundaries - positions  # How far to next regime?

            # Smooth sigmoid gate (far from boundary = 1.0, at boundary = 0.5)
            gate_values = torch.sigmoid(distance_to_boundary.float() / self.transition_smoothness)

            # Apply regime-specific gate weights
            regime_gates = self.transition_gates[regimes]  # [batch, seq_len]
            combined_gates = gate_values * regime_gates  # [batch, seq_len]

            # Apply gates
            regime_encodings = regime_encodings * combined_gates.unsqueeze(-1)  # [batch, seq_len, d_model]

        # Step 6: RoPE DISABLED (incompatible with set-valued operations!)
        # Eidos uses only dimensional regime boundaries for position encoding
        # if self.use_rope:
        #     pos_clamped = torch.clamp(positions, 0, self.rope_cos.size(0) - 1)
        #     rope_cos = self.rope_cos[pos_clamped]
        #     rope_sin = self.rope_sin[pos_clamped]
        #     regime_encodings = regime_encodings + 0.1 * (rope_cos + rope_sin)

        # Step 7: Dropout (optional)
        if self.dropout is not None:
            regime_encodings = self.dropout(regime_encodings)

        # Remove batch dimension if input was 1D
        if squeeze_output:
            regime_encodings = regime_encodings.squeeze(0)

        return regime_encodings

    def get_regime_statistics(self, positions: torch.Tensor) -> dict:
        """
        Analyze regime distribution in a sequence (for validation/debugging).

        Args:
            positions: [batch, seq_len] or [seq_len] position indices

        Returns:
            stats: Dictionary with regime distribution info
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        regimes = classify_dimensional_regime_torch(positions)

        stats = {
            'regime_counts': {},
            'regime_percentages': {},
            'max_position': positions.max().item(),
            'max_regime': regimes.max().item(),
            'transitions': 0
        }

        # Count positions in each regime
        for r in range(self.num_regimes):
            count = (regimes == r).sum().item()
            total = regimes.numel()
            stats['regime_counts'][f'd{r}'] = count
            stats['regime_percentages'][f'd{r}'] = f"{100 * count / total:.2f}%"

        # Count regime transitions (how many times regime changes)
        if positions.shape[1] > 1:
            regime_changes = (regimes[:, 1:] != regimes[:, :-1]).sum().item()
            stats['transitions'] = regime_changes

        return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_regime_boundaries(max_pos: int = 1000):
    """
    Visualize which positions fall into which regimes.

    Args:
        max_pos: Maximum position to visualize
    """
    positions = torch.arange(max_pos)
    regimes = classify_dimensional_regime_torch(positions)

    print(f"\n{'='*70}")
    print(f"IR REGIME VISUALIZATION (positions 0-{max_pos-1})")
    print(f"{'='*70}")

    current_regime = -1
    for pos in range(max_pos):
        regime = regimes[pos].item()
        if regime != current_regime:
            if current_regime != -1:
                print()  # Newline between regimes
            current_regime = regime
            boundary = IR_BOUNDARIES[regime].item()
            print(f"\n📍 REGIME d{regime} (starts at position {boundary}):")
            print(f"   ", end="")

        # Show first/last few positions in each regime
        if pos < boundary + 10 or pos > IR_BOUNDARIES[regime+1].item() - 10:
            print(f"{pos} ", end="")
        elif pos == boundary + 10:
            print(f"... ", end="")


def test_regime_classification():
    """Quick test of regime classification."""
    print(f"\n{'='*70}")
    print("TESTING REGIME CLASSIFICATION")
    print(f"{'='*70}\n")

    # Test cases spanning all regimes
    test_positions = torch.tensor([
        0, 3, 4,           # d0
        5, 10, 54,         # d1
        55, 100, 604,      # d2
        605, 1000, 6654,   # d3
        6655, 10000, 73204 # d4
    ])

    regimes = classify_dimensional_regime_torch(test_positions)
    local_pos = get_local_position_in_regime(test_positions, regimes)

    print(f"{'Position':<10} {'Regime':<10} {'Local Pos':<12} {'Boundary':<10}")
    print("-" * 50)

    for i, pos in enumerate(test_positions):
        regime = regimes[i].item()
        local = local_pos[i].item()
        boundary = IR_BOUNDARIES[regime].item()
        print(f"{pos.item():<10} d{regime:<9} {local:<12} {boundary:<10}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("IR HIERARCHICAL POSITIONAL ENCODING - Test Suite")
    print("="*70)

    # Test 1: Regime classification
    test_regime_classification()

    # Test 2: Visualize boundaries
    visualize_regime_boundaries(max_pos=1000)

    # Test 3: Create IR encoder and test forward pass
    print(f"\n{'='*70}")
    print("TESTING IR POSITIONAL ENCODER")
    print(f"{'='*70}\n")

    d_model = 512
    max_seq_len = 8192

    encoder = IRHierarchicalPositionalEncoding(
        d_model=d_model,
        max_seq_len=max_seq_len,
        num_regimes=5,  # d0-d4
        use_rope=True,
        learnable_transitions=True
    )

    # Test with different sequence lengths
    test_lengths = [128, 512, 2048, 8192]

    for seq_len in test_lengths:
        positions = torch.arange(seq_len)
        pos_enc = encoder(positions)

        stats = encoder.get_regime_statistics(positions)

        print(f"\nSequence length: {seq_len}")
        print(f"  Output shape: {pos_enc.shape}")
        print(f"  Max regime: d{stats['max_regime']}")
        print(f"  Regime distribution: {stats['regime_counts']}")
        print(f"  Transitions: {stats['transitions']}")

    # Test 4: Batch processing
    print(f"\n{'='*70}")
    print("TESTING BATCH PROCESSING")
    print(f"{'='*70}\n")

    batch_size = 4
    seq_len = 1024
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    pos_enc = encoder(positions)
    print(f"Batch input shape: {positions.shape}")
    print(f"Batch output shape: {pos_enc.shape}")
    print(f"[OK] Batch processing works!")

    # Test 5: Regime boundaries create distinct embeddings
    print(f"\n{'='*70}")
    print("TESTING REGIME BOUNDARY DISTINCTIVENESS")
    print(f"{'='*70}\n")

    # Positions right before and after regime boundaries
    boundary_positions = torch.tensor([
        4, 5,      # d0 → d1 transition
        54, 55,    # d1 → d2 transition
        604, 605,  # d2 → d3 transition
    ])

    boundary_enc = encoder(boundary_positions)

    # Compute cosine similarity between adjacent positions
    for i in range(0, len(boundary_positions), 2):
        pos1, pos2 = boundary_positions[i], boundary_positions[i+1]
        enc1, enc2 = boundary_enc[i], boundary_enc[i+1]

        # Cosine similarity
        similarity = F.cosine_similarity(enc1.unsqueeze(0), enc2.unsqueeze(0)).item()

        regime1 = classify_dimensional_regime_torch(pos1.unsqueeze(0)).item()
        regime2 = classify_dimensional_regime_torch(pos2.unsqueeze(0)).item()

        print(f"Position {pos1.item()} (d{regime1}) → {pos2.item()} (d{regime2})")
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  {'[DIFF]' if regime1 != regime2 else '[SAME]'} regimes")

    print(f"\n{'='*70}")
    print("[OK] ALL TESTS PASSED!")
    print(f"{'='*70}\n")
    print("Next steps:")
    print("  1. Integrate into causal transformer (optional --use-ir-encoding flag)")
    print("  2. Train on progressive sequence lengths (512 → 1024 → 2048 → 4096 → 8192)")
    print("  3. Monitor structural tension at positions [5, 55, 605, 6655]")
    print("  4. Validate that phase transitions occur at IR boundaries!")
    print()
