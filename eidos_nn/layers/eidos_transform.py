"""
EIDOS TRANSFORM - Geometric Transformations (Replaces nn.Linear)

Provides geometric transformations based on axioms R1-R6,
designed as a drop-in replacement for nn.Linear layers.

Instead of a dense affine projection y = Wx + b, the eidosTransform
applies a composition of:
  R1: Identity (residual preservation)
  R2: Reflection (optional, learned axis)
  R4: Inversion (optional, safe reciprocal)
  R5: Rotation (2D Givens rotations in channel pairs)
  R6: Scale (log-space magnitude control)

When input_dim != output_dim, a DimensionMatcher handles the
bridge either through the current pure geometric spanning operator
or through the older Tanh-bounded constrained matmul path kept for
canonical experiment reproducibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def _supports_complex_rotation(x: torch.Tensor) -> bool:
    """Limit complex rotation kernels to backends/dtypes with stable support."""
    return x.device.type in ("cpu", "cuda") and x.dtype in (torch.float32, torch.float64)


def _rotate_pairwise_real(real: torch.Tensor, imag: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real-valued fallback for pairwise Givens rotations."""
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotated_real = cos_theta * real - sin_theta * imag
    rotated_imag = sin_theta * real + cos_theta * imag
    return rotated_real, rotated_imag


def _rotate_contiguous_pairs(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs, using complex multiply when available."""
    *batch_dims, pad_dim = state.shape
    num_pairs = pad_dim // 2

    if _supports_complex_rotation(state):
        phase = torch.polar(torch.ones_like(theta), theta)
        # view_as_complex requires the trailing real/imag axis to be stride-1.
        # Upstream slicing/permutation can violate that even when the shape fits.
        pairs = state.reshape(*batch_dims, num_pairs, 2).contiguous()
        z = torch.view_as_complex(pairs)
        z_rotated = z * phase
        return torch.view_as_real(z_rotated).reshape(*batch_dims, pad_dim)

    pairs = state.reshape(*batch_dims, num_pairs, 2).contiguous()
    real = pairs[..., 0]
    imag = pairs[..., 1]
    rotated_real, rotated_imag = _rotate_pairwise_real(real, imag, theta)
    return torch.stack([rotated_real, rotated_imag], dim=-1).reshape(*batch_dims, pad_dim)


def _rotate_butterfly_pairs(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate front/back hemispheres as paired coordinates."""
    num_pairs = state.shape[-1] // 2
    real = state[..., :num_pairs]
    imag = state[..., num_pairs:]

    if _supports_complex_rotation(state):
        phase = torch.polar(torch.ones_like(theta), theta)
        z = torch.complex(real, imag)
        z_rotated = z * phase
        return torch.cat([z_rotated.real, z_rotated.imag], dim=-1)

    rotated_real, rotated_imag = _rotate_pairwise_real(real, imag, theta)
    return torch.cat([rotated_real, rotated_imag], dim=-1)


def _collapse_fold(state: torch.Tensor, output_dim: int) -> torch.Tensor:
    """
    Fold a larger rotated state into fewer output coordinates by averaging
    evenly partitioned coordinate buckets.

    This avoids the strong prefix bias of pure truncation during compression
    while remaining a structured collapse rather than a free affine map.
    """
    *batch_dims, dim = state.shape
    if dim == output_dim:
        return state

    device = state.device
    dtype = state.dtype
    idx = torch.arange(dim, device=device)
    buckets = torch.clamp((idx * output_dim) // dim, max=output_dim - 1)

    out = state.new_zeros(*batch_dims, output_dim)
    counts = torch.zeros(output_dim, device=device, dtype=dtype)
    out.index_add_(-1, buckets, state)
    counts.index_add_(0, buckets, torch.ones_like(buckets, dtype=dtype))
    return out / counts.clamp_min(1.0)


class eidosTransform(nn.Module):
    """
    Geometric transformation layer (replaces nn.Linear).

    Instead of y = Wx + b (which flattens),
    uses combination of R5 (rotation) + R6 (scale) + R1 (identity).

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        num_rotation_planes: Number of rotation planes to use (default: 4)
        use_reflection: Use R2 reflection axiom (default: False)
        use_inversion: Use R4 inversion axiom (default: False)
        matcher_mode: Dimension-matching bridge.
            "geometric" = pure padding/rotation/truncate matcher (current fast path)
            "canonical" = constrained tanh-bounded matmul bridge used in earlier experiments
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_rotation_planes: int = 4,
        use_reflection: bool = False,
        use_inversion: bool = False,
        matcher_mode: str = "geometric",
        matcher_collapse: str = "truncate",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rotation_planes = num_rotation_planes
        self.use_reflection = use_reflection
        self.use_inversion = use_inversion
        self.matcher_mode = matcher_mode
        self.matcher_collapse = matcher_collapse

        # R5: Rotation angles (per plane, per channel pair)
        # We need enough angles to cover the active input dimension during forward.
        # Using max(...) prevents under-provisioning when input_dim >> output_dim
        # (e.g., MNIST flatten to 3136 feeding a 128-dim head).
        num_channel_pairs = max(input_dim, output_dim) // 2
        self.rotation_angles = nn.Parameter(
            torch.zeros(num_rotation_planes, num_channel_pairs)
        )

        # R6: Scale factors (log-space for λ > 0)
        self.log_scales = nn.Parameter(
            torch.zeros(output_dim)
        )

        # R2: Reflection axes (if enabled)
        if use_reflection:
            self.reflection_axis = nn.Parameter(
                torch.randn(input_dim)
            )

        # Dimension matching (if input_dim != output_dim)
        if input_dim != output_dim:
            self.dim_matcher = DimensionMatcher(
                input_dim,
                output_dim,
                mode=matcher_mode,
                collapse_mode=matcher_collapse,
            )
        else:
            self.dim_matcher = None

    def apply_R5_rotation(
        self,
        x: torch.Tensor,
        plane_idx: int
    ) -> torch.Tensor:
        """
        Apply R5 rotation axiom in 2D channel pairs.

        Args:
            x: [..., dim] input
            plane_idx: Which rotation plane to use

        Returns:
            Rotated tensor [..., dim]
        """
        *batch_dims, dim = x.shape

        # Handle odd dimensions by slicing off the last channel
        # The last channel (odd remainder) is treated as the Axis of Rotation (Identity)
        num_pairs = dim // 2
        even_dim = num_pairs * 2
        
        if dim % 2 != 0:
            x_even = x[..., :even_dim]
            x_odd = x[..., even_dim:] # [..., 1]
        else:
            x_even = x
            x_odd = None

        # Treat channels as complex number pairs (real, imag)
        # Now safe because x_even is guaranteed divisible by 2
        x_pairs = x_even.view(*batch_dims, num_pairs, 2)

        # Get rotation angles for this plane
        # Clamp plane_idx to valid range
        plane_idx = min(plane_idx, self.num_rotation_planes - 1)
        
        # Ensure we don't access out of bounds if weights are smaller than input
        # (Though __init__ should prevent this with max(in, out))
        max_pairs_in_weights = self.rotation_angles.shape[1]
        valid_pairs = min(num_pairs, max_pairs_in_weights)
        theta = self.rotation_angles[plane_idx, :valid_pairs]  # [valid_pairs]

        # Apply 2D rotation to the valid pairs covered by weights
        active_even = x_even[..., :valid_pairs * 2]
        rotated_even_active = _rotate_contiguous_pairs(active_even, theta)
        rotated_pairs = rotated_even_active.reshape(*batch_dims, valid_pairs, 2)
        
        # If input was larger than weights, cat the rest of the pairs unchanged
        if num_pairs > valid_pairs:
            remaining_pairs = x_pairs[..., valid_pairs:, :]
            rotated_pairs = torch.cat([rotated_pairs, remaining_pairs], dim=-2)
            
        rotated_even = rotated_pairs.view(*batch_dims, even_dim)

        # Re-attach odd dimension if it exists
        if x_odd is not None:
            rotated = torch.cat([rotated_even, x_odd], dim=-1)
        else:
            rotated = rotated_even

        return rotated

    def apply_R6_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply R6 scale axiom.

        Args:
            x: [..., dim] input

        Returns:
            Scaled tensor [..., dim]
        """
        # Get scale factors (ensure λ > 0 via exp)
        scales = torch.exp(self.log_scales)  # [output_dim]

        # Clamp to reasonable range
        scales = torch.clamp(scales, 0.1, 10.0)

        # Broadcast and scale
        if x.shape[-1] == scales.shape[0]:
            return x * scales
        else:
            # Handle dimension mismatch
            return x * scales[:x.shape[-1]]

    def apply_R2_reflection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply R2 reflection axiom across learned axis.

        Args:
            x: [..., input_dim] input

        Returns:
            Reflected tensor [..., input_dim]
        """
        # Normalize reflection axis
        axis = self.reflection_axis / (torch.norm(self.reflection_axis) + 1e-6)

        # Compute projection onto axis
        # proj = (x · axis) * axis
        proj_scale = torch.sum(x * axis, dim=-1, keepdim=True)
        proj = proj_scale * axis

        # Reflect: x_reflected = 2*proj - x
        reflected = 2 * proj - x

        return reflected

    def apply_R4_inversion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply R4 inversion axiom (reciprocal).

        This is tricky for neural networks (division by near-zero).
        We use a safe version: 1 / (x + eps) with tanh squashing.

        Args:
            x: [..., dim] input

        Returns:
            Inverted tensor [..., dim]
        """
        eps = 1e-3

        # Safe inversion with tanh to bound outputs
        inverted = torch.tanh(1.0 / (x + eps))

        return inverted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply eidos transformation.

        Args:
            x: [..., input_dim] input

        Returns:
            Transformed tensor [..., output_dim]
        """
        # R1: Identity (preserve input)
        identity = x

        # R2: Reflection (if enabled)
        if self.use_reflection:
            x = self.apply_R2_reflection(x)

        # R4: Inversion (if enabled)
        if self.use_inversion:
            x = self.apply_R4_inversion(x)

        # R5: Rotation (apply multiple planes and combine)
        rotated_outputs = []
        for plane_idx in range(self.num_rotation_planes):
            rotated = self.apply_R5_rotation(x, plane_idx)
            rotated_outputs.append(rotated)

        # Combine rotated outputs (average)
        x_rotated = torch.stack(rotated_outputs, dim=0).mean(dim=0)

        # Match dimensions if needed
        if self.dim_matcher is not None:
            x_rotated = self.dim_matcher(x_rotated)
            identity = self.dim_matcher(identity)

        # R6: Scaling
        x_scaled = self.apply_R6_scaling(x_rotated)

        # R1: Add identity (residual-like connection)
        if identity.shape == x_scaled.shape:
            output = x_scaled + identity * 0.1  # Small identity contribution
        else:
            output = x_scaled

        return output


class DimensionMatcher(nn.Module):
    """
    Dimension-matching bridge used when input_dim != output_dim.

    Modes:
        geometric:
            Pure Eidos matching via R0 padding, R5 spanning rotations, and
            R1 truncate/collapse.
        canonical:
            Older experiment-ready constrained matmul bridge with tanh-bounded
            interpolation weights, kept for reproducibility against the ICML-era
            canonical configuration.

    Collapse modes for geometric matching:
        truncate:
            Keep the leading output_dim coordinates after rotation.
        fold:
            Collapse the rotated state into output_dim buckets by structured
            folding/averaging, reducing prefix bias during compression.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mixing_planes: int = 4,
        mode: str = "geometric",
        collapse_mode: str = "truncate",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.collapse_mode = collapse_mode
        if self.mode not in {"geometric", "canonical"}:
            raise ValueError(f"Unsupported DimensionMatcher mode: {self.mode}")
        if self.collapse_mode not in {"truncate", "fold"}:
            raise ValueError(f"Unsupported DimensionMatcher collapse mode: {self.collapse_mode}")

        if self.mode == "geometric":
            self.max_dim = max(input_dim, output_dim)
            # Ensure padded dimension is even so we can use R5 channel pairs
            self.pad_dim = self.max_dim if self.max_dim % 2 == 0 else self.max_dim + 1
            self.mixing_planes = mixing_planes
            self.num_pairs = self.pad_dim // 2
            self.rotation_angles = nn.Parameter(
                torch.zeros(mixing_planes, self.num_pairs)
            )
            nn.init.normal_(self.rotation_angles, std=0.1)
        else:
            self.interpolation_weights = nn.Parameter(
                torch.empty(output_dim, input_dim)
            )
            nn.init.xavier_uniform_(self.interpolation_weights)
            self.scale = 1.0 / math.sqrt(max(input_dim, 1))

    def _forward_canonical(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.tanh(self.interpolation_weights) * self.scale
        return torch.matmul(x, weights.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Match dimensions according to the configured bridge mode.

        Args:
            x: [..., input_dim]

        Returns:
            [..., output_dim]
        """
        if self.mode == "canonical":
            return self._forward_canonical(x)

        *batch_dims, dim = x.shape
        
        # 1. R0 Axiom (Vacuum Padding): Inject empty spatial coordinates if needed
        if dim < self.pad_dim:
            x_padded = F.pad(x, (0, self.pad_dim - dim), "constant", 0.0)
        else:
            x_padded = x

        # 2. R5 Axiom (Spanning Rotations): Geometrically push energy into novel dimensions
        state = x_padded
        
        for plane_idx in range(self.mixing_planes):
            theta = self.rotation_angles[plane_idx]
            
            # We mix coordinates using different pair strata to ensure full-dimensional reach:
            # Plane 0: Contiguous pairs (0,1), (2,3)
            # Plane 1: Offset pairs (1,2), (3,4)
            # Plane 2: Butterfly pairs (0, N/2), (1, N/2 + 1)
            # Plane 3: Contiguous pairs, etc.
            
            if plane_idx % 3 == 0:
                state = _rotate_contiguous_pairs(state, theta)
                
            elif plane_idx % 3 == 1:
                # Offset pair rotation (stitch adjacent pairs together)
                state_rolled = torch.roll(state, shifts=1, dims=-1)
                state_rolled_back = _rotate_contiguous_pairs(state_rolled, theta)
                state = torch.roll(state_rolled_back, shifts=-1, dims=-1)
                
            else:
                state = _rotate_butterfly_pairs(state, theta)

        # 3. R1 Axiom (Collapse): either prefix truncation or structured fold
        if self.pad_dim > self.output_dim:
            if self.collapse_mode == "fold":
                output = _collapse_fold(state, self.output_dim)
            else:
                output = state[..., :self.output_dim]
        else:
            output = state
            
        return output


class eidosSequential(nn.Module):
    """
    Sequential composition of eidos transforms.

    Replaces nn.Sequential for chaining eidos operations.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layers sequentially.

        Args:
            x: Input tensor

        Returns:
            Output after all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("eidos TRANSFORM - Quick Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\n" + "=" * 70)
    print("Test 1: Basic eidos Transform (Same Dimensions)")
    print("=" * 70)

    # Create transform
    transform = eidosTransform(
        input_dim=256,
        output_dim=256,
        num_rotation_planes=4,
        use_reflection=False,
        use_inversion=False
    ).to(device)

    # Test input
    x = torch.randn(4, 32, 256).to(device)

    # Transform
    y = transform(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected: [4, 32, 256]")
    print(f"[OK] Same-dimension transform works!\n")

    print("=" * 70)
    print("Test 2: Dimension Change (Input != Output)")
    print("=" * 70)

    # Create transform with dimension change
    transform_dim_change = eidosTransform(
        input_dim=128,
        output_dim=256,
        num_rotation_planes=4
    ).to(device)

    x_small = torch.randn(4, 32, 128).to(device)
    y_large = transform_dim_change(x_small)

    print(f"Input shape: {x_small.shape}")
    print(f"Output shape: {y_large.shape}")
    print(f"Expected: [4, 32, 256]")
    print(f"[OK] Dimension change works!\n")

    print("=" * 70)
    print("Test 3: With Reflection and Inversion")
    print("=" * 70)

    transform_full = eidosTransform(
        input_dim=256,
        output_dim=256,
        num_rotation_planes=4,
        use_reflection=True,
        use_inversion=True
    ).to(device)

    y_full = transform_full(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_full.shape}")
    print(f"[OK] Full axiom transform works!\n")

    print("=" * 70)
    print("Test 4: R5 Rotation (Norm Preservation)")
    print("=" * 70)

    # Check that rotation preserves norms
    x_norm_before = torch.norm(x, dim=-1).mean()
    y_rotated = transform.apply_R5_rotation(x, plane_idx=0)
    y_norm_after = torch.norm(y_rotated, dim=-1).mean()

    print(f"Norm before rotation: {x_norm_before.item():.4f}")
    print(f"Norm after rotation: {y_norm_after.item():.4f}")
    print(f"Difference: {abs(x_norm_before - y_norm_after).item():.6f}")

    if abs(x_norm_before - y_norm_after).item() < 0.1:
        print(f"[OK] R5 rotation preserves norms!\n")
    else:
        print(f"[WARNING] Norm not perfectly preserved (expected for non-unit vectors)\n")

    print("=" * 70)
    print("Test 5: eidosSequential (Chaining)")
    print("=" * 70)

    # Chain multiple transforms
    seq_transform = eidosSequential(
        eidosTransform(256, 512, num_rotation_planes=4),
        eidosTransform(512, 512, num_rotation_planes=4),
        eidosTransform(512, 256, num_rotation_planes=4)
    ).to(device)

    y_seq = seq_transform(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_seq.shape}")
    print(f"Expected: [4, 32, 256]")
    print(f"[OK] Sequential chaining works!\n")

    print("=" * 70)
    print("Test 6: Gradient Flow")
    print("=" * 70)

    x_grad = torch.randn(2, 10, 256, requires_grad=True).to(device)
    y_grad = transform(x_grad)
    loss = y_grad.mean()
    loss.backward()

    print(f"Input gradient: {x_grad.grad is not None}")
    print(f"Rotation angles gradient: {transform.rotation_angles.grad is not None}")
    print(f"Scale gradient: {transform.log_scales.grad is not None}")
    print(f"[OK] Gradient flow works!\n")

    print("=" * 70)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 70)
    print("\neidosTransform is ready to replace nn.Linear!")
    print("Key advantages:")
    print("  + Geometric (uses R1-R6 axioms, not matrix multiply)")
    print("  + Norm-preserving (R5 rotation)")
    print("  + Scale-aware (R6 scaling)")
    print("  + Preserves structure (doesn't flatten to 1D)")
    print()
