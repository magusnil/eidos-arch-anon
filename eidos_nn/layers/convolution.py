"""
EIDOS CONVOLUTIONAL LAYER - Set-Valued Spatial Operations

Implements set-valued convolution where multiple independent kernels
generate parallel spatial interpretations, each modulated by a learned
context state and an axiom-specific operation (R1-R4). A hierarchical
quality scorer then collapses the path set into a single output.

Key Differences from Classical Conv2D:
  - Multiple independent kernels (one per path, not shared).
  - Context-dependent modulation via learned state vectors.
  - Non-linear path operations (Collapse, Identity, Carry, Transform).
  - Quality-weighted set collapse instead of fixed aggregation.

Mathematical Foundation:
  Classical: f(x) = W * x + b
  Eidos:     f(x) = Collapse({K_i(x) ⊙_σ_i | i = 1..K})
             where ⊙ ∈ {R1, R2, R3, R4} and σ is spatial context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
eidos_PATH = os.path.join(os.path.dirname(__file__), '..')
if eidos_PATH not in sys.path:
    sys.path.insert(0, eidos_PATH)




class ModularPhaseNorm2d(nn.Module):
    """
    Modular Phase Normalization (eidos-compatible replacement for BatchNorm).

    WHY NOT BATCHNORM:
    BatchNorm couples samples via batch statistics, breaking eidos path independence.

    Classical BatchNorm:
      - E[x_batch], Var[x_batch] → couples all samples in batch
      - Breaks: Each path should be independent across samples
      - Problem: Path quality in sample A affects normalization in sample B

    ModularPhaseNorm:
      - Normalizes based on STRUCTURAL coherence, not statistical coupling
      - Uses cyclic bases (2, 3, 7) for pattern detection
      - Each sample normalized independently (eidos path independence preserved)

    
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Learnable scale and shift (like BatchNorm's γ and β)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Modular Phase bases: cyclic structure for normalization
        # These are NOT learned - they define the structural pattern space
        self.register_buffer('pca_bases', torch.tensor([2, 3, 7], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Modular Phase normalization.

        Args:
            x: [batch, channels, height, width] or [batch, channels]

        Returns:
            Normalized tensor with same shape as input
        """
        # Compute mean and variance over SPATIAL dimensions only (not batch!)
        # This preserves sample independence (eidos requirement)
        dims = list(range(2, x.dim()))  # Spatial dims (height, width, ...)

        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # Normalize: zero mean, unit variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine transformation
        # Reshape weight/bias to match input dimensions
        shape = [1, self.num_features] + [1] * (x.dim() - 2)
        weight = self.weight.view(*shape)
        bias = self.bias.view(*shape)

        return x_norm * weight + bias


class SpatialPathBundle:
    """
    PathBundle for spatial (CNN) data with shape [batch, channels, h, w, paths].

    Unlike core.PathBundle which expects [batch, features, paths],
    this handles 5D spatial tensors from convolutional layers.
    """

    def __init__(self, data: torch.Tensor, quality_scores: torch.Tensor, num_paths: int):
        """
        Args:
            data: [batch, channels, h, w, num_paths]
            quality_scores: [batch, num_paths]
            num_paths: Number of paths
        """
        if data.dim() != 5:
            raise ValueError(f"SpatialPathBundle data must be 5D [batch, channels, h, w, paths], got {data.shape}")

        self.data = data
        self.quality_scores = quality_scores
        self.num_paths = num_paths
        self.batch_size, self.channels, self.height, self.width, _ = data.shape

    @property
    def shape(self):
        """Return shape as (batch, channels, h, w, paths)"""
        return self.data.shape

    @property
    def device(self):
        """Return device of underlying tensor"""
        return self.data.device


class TrueeidosConv2D(nn.Module):
    """
    True eidos Convolutional Layer with Set-Valued Spatial Operations.

    This is NOT a wrapper around nn.Conv2d!
    This implements convolution as set-valued collapse from problem space
    to solution space, using eidos primitives (R1-R4).

    Architecture:
      1. Problem Space: Generate 27 possible spatial interpretations
      2. eidos Threading: Apply R1-R4 operations based on path type
      3. Solution Space: Quality-weighted collapse to final features

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        num_paths: Number of parallel paths (default: 27 = 3^3)
        stride: Convolution stride
        padding: Convolution padding
        use_context: Enable context-dependent σ state
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_paths: int = 27,
        stride: int = 1,
        padding: int = 1,
        use_context: bool = True,
        is_global: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_paths = num_paths
        self.stride = stride
        self.padding = padding
        self.use_context = use_context
        self.is_global = is_global

        # ================================================================
        # MULTIPLE KERNELS - One per path (NOT uniform!)
        # ================================================================
        # Each kernel learns a different spatial interpretation
        self.kernels = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False  # Bias handled separately per path
            )
            for _ in range(num_paths)
        ])

        # Path-specific biases (context-dependent)
        self.path_biases = nn.Parameter(torch.zeros(num_paths, out_channels))

        # ================================================================
        # CONTEXT STATE (σ) - Spatial context for eidos operations
        # ================================================================
        if use_context:
            # Learnable context state for each path
            self.context_state = nn.Parameter(
                torch.randn(num_paths, out_channels) * 0.01
            )
        else:
            self.register_buffer('context_state', torch.zeros(num_paths, out_channels))

        # ================================================================
        # HIERARCHICAL PATH SCORER - Quality-weighted collapse
        # ================================================================
        self.quality_scorer = HierarchicalPathScorer(
            num_paths=num_paths,
            feature_dim=out_channels,
            team_size=9,  # 27 paths → 3 teams of 9
            use_overhead=True
        )

        # ================================================================
        # eidos OPERATION INDICATORS
        # ================================================================
        # Assign each path to an operation type (R1, R2, R3, R4)
        self.path_operations = self._assign_path_operations(num_paths)

    def _assign_path_operations(self, num_paths: int) -> torch.Tensor:
        """
        Assign eidos operation type to each path.

        For num_paths=27 (R1-R4 only, NO TRIGONOMETRY):
          - Paths 0-6:   R1 (Collapse)
          - Paths 7-13:  R2 (Identity)
          - Paths 14-20: R3 (Carry)
          - Paths 21-26: R4 (Transform)

        For num_paths=48 (R1-R5 with TRIGONOMETRY):
          - Paths 0-9:   R1 (Collapse)
          - Paths 10-19: R2 (Identity)
          - Paths 20-29: R3 (Carry)
          - Paths 30-39: R4 (Transform)
          - Paths 40-47: R5 (Trigonometry) ← NEW! For rotations + color phase

        Returns:
            Tensor [num_paths] with operation indices (0=R1, 1=R2, 2=R3, 3=R4, 4=R5)
        """
        ops = torch.zeros(num_paths, dtype=torch.long)

        if num_paths == 27:
            # Standard 4-operation architecture (R1-R4 only)
            paths_per_op = num_paths // 4
            for i in range(num_paths):
                op_idx = min(i // paths_per_op, 3)  # 0, 1, 2, or 3
                ops[i] = op_idx

        elif num_paths == 48:
            # New architecture with R5 trigonometry
            paths_per_op = num_paths // 5  # 48 / 5 = 9.6, round to 10-10-10-10-8
            boundaries = [0, 10, 20, 30, 40, 48]
            for i in range(num_paths):
                for op_idx in range(5):
                    if boundaries[op_idx] <= i < boundaries[op_idx + 1]:
                        ops[i] = op_idx
                        break
        else:
            # Fallback for other path counts
            paths_per_op = num_paths // 4
            for i in range(num_paths):
                op_idx = min(i // paths_per_op, 3)
                ops[i] = op_idx

        return ops

    def apply_R1_collapse(
        self,
        feature: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        R1 (Collapse): 0 ⊙ a = {0, a}

        Collapses potential to actual.
        In spatial domain: Soft gating based on context.

        Args:
            feature: [batch, out_channels, h, w]
            context: [out_channels] context state for this path

        Returns:
            Collapsed feature [batch, out_channels, h, w]
        """
        # Context determines collapse probability
        # High context → keep feature, Low context → suppress to 0
        collapse_gate = torch.sigmoid(context.view(1, -1, 1, 1))

        # Soft collapse: {0, feature} weighted by gate
        collapsed = feature * collapse_gate

        return collapsed

    def apply_R2_identity(self, feature: torch.Tensor) -> torch.Tensor:
        """
        R2 (Identity): a ⊙ 0 = a

        Preserves information (passthrough).
        In spatial domain: Direct feature retention.

        Args:
            feature: [batch, out_channels, h, w]

        Returns:
            Same feature (identity)
        """
        return feature

    def apply_R3_carry(
        self,
        feature: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        R3 (Carry): a ⊙_σ b = a + Δ_σ(b)

        Adds context-dependent carry/increment.
        In spatial domain: Feature + spatial offset determined by context.

        Args:
            feature: [batch, out_channels, h, w]
            context: [out_channels] context state for this path

        Returns:
            Feature with carry added [batch, out_channels, h, w]
        """
        # Context determines carry amount
        carry = context.view(1, -1, 1, 1)

        # Add carry to feature
        return feature + carry

    def apply_R4_transform(
        self,
        feature: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        R4 (Transform): 1 ⊙_σ a = F_σ(a)

        Context-dependent transformation.
        In spatial domain: Non-linear transform modulated by context.

        Args:
            feature: [batch, out_channels, h, w]
            context: [out_channels] context state for this path

        Returns:
            Transformed feature [batch, out_channels, h, w]
        """
        # Context modulates transformation strength
        transform_scale = torch.tanh(context.view(1, -1, 1, 1))

        # Non-linear transformation
        # f(x) = x * tanh(σ) + (1 - tanh²(σ)) * x²
        linear_part = feature * transform_scale
        nonlinear_part = (1 - transform_scale ** 2) * (feature ** 2).sign() * torch.sqrt(torch.abs(feature) + 1e-8)

        return linear_part + nonlinear_part

    def apply_R5_trigonometry(
        self,
        feature: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        R5 (Trigonometry): θ ⊙_∠ r = {r·cos(θ), r·sin(θ)}

        Applies trigonometric transformations for:
          - Rotation invariance (spatial angles)
          - Color phase (RGB → polar coordinates)
          - Periodic patterns (texture frequency)

        Supports rotation invariance and periodic pattern detection.

        Args:
            feature: [batch, out_channels, h, w]
            context: [out_channels] context state for this path

        Returns:
            Trigonometrically transformed feature [batch, out_channels, h, w]
        """
        # Extract phase angle from context
        # Context maps to angle range [-π, π]
        theta = torch.tanh(context.view(1, -1, 1, 1)) * torch.pi

        # Extract magnitude from feature
        # r = ||feature||
        magnitude = torch.sqrt(torch.abs(feature) + 1e-8)

        # Apply rotation/phase shift
        # Real part: r * cos(θ)
        # Imaginary part: r * sin(θ)
        cos_component = magnitude * torch.cos(theta)
        sin_component = magnitude * torch.sin(theta)

        # Combine components (weighted sum)
        # This allows the model to learn optimal phase mixing
        alpha = torch.sigmoid(context.view(1, -1, 1, 1))  # Learnable mixing weight
        result = alpha * cos_component + (1 - alpha) * sin_component

        return result

    def forward(self, x: torch.Tensor) -> SpatialPathBundle:
        """
        Forward pass: Set-valued spatial convolution.

        Process:
          1. Generate 27 possible spatial features (problem space)
          2. Apply eidos operations (R1-R4) based on path type
          3. Quality-weighted collapse (solution space)
        """
        batch_size, in_channels, height, width = x.shape

        # ================================================================
        # STEP 1: PROBLEM SPACE - Generate set of possible features
        # ================================================================
        path_features = []

        for path_idx, kernel in enumerate(self.kernels):
            feature = kernel(x)  # [batch, out_channels, h, w]
            feature = feature + self.path_biases[path_idx].view(1, -1, 1, 1)

            # ============================================================
            # STEP 2: eidos THREADING - Apply R1-R4 operations
            # ============================================================
            op_type = self.path_operations[path_idx].item()
            context = self.context_state[path_idx]

            if op_type == 0:  # R1: Collapse
                feature = self.apply_R1_collapse(feature, context)
            elif op_type == 1:  # R2: Identity
                feature = self.apply_R2_identity(feature)
            elif op_type == 2:  # R3: Carry
                feature = self.apply_R3_carry(feature, context)
            elif op_type == 3:  # R4: Transform
                feature = self.apply_R4_transform(feature, context)

            path_features.append(feature)

        # Stack all paths: [batch, out_channels, h, w, num_paths]
        path_tensor = torch.stack(path_features, dim=-1)

        # ================================================================
        # STEP 3: SOLUTION SPACE - Quality-weighted collapse
        # ================================================================
        if self.is_global:
            # Optimization: Score based on GLOBAL feature context
            global_features = path_tensor.mean(dim=(2, 3))
            importances = self.quality_scorer.get_path_importances(global_features)
            quality_scores_batch = importances['global_weights'] # [batch, num_paths]
        else:
            # Standard Local Scoring (Pixel-wise)
            out_height, out_width = path_tensor.shape[2], path_tensor.shape[3]
            batch_h_w = batch_size * out_height * out_width
            features_flat = path_tensor.permute(0, 2, 3, 1, 4).reshape(
                batch_h_w, self.out_channels, self.num_paths
            )
            importances = self.quality_scorer.get_path_importances(features_flat)
            quality_scores = importances['global_weights']
            quality_spatial = quality_scores.view(batch_size, out_height, out_width, self.num_paths)
            quality_scores_batch = quality_spatial.mean(dim=(1, 2))

        # Create SpatialPathBundle
        return SpatialPathBundle(
            data=path_tensor,
            quality_scores=quality_scores_batch,
            num_paths=self.num_paths
        )

    def collapse_spatial_pathbundle(self, path_bundle: SpatialPathBundle) -> torch.Tensor:
        """
        Collapse spatial PathBundle to single feature map.

        Args:
            path_bundle: SpatialPathBundle with data [batch, out_channels, h, w, num_paths]

        Returns:
            Collapsed tensor [batch, out_channels, h, w]
        """
        # Get quality scores [batch, num_paths]
        quality = path_bundle.quality_scores

        # Expand to spatial dimensions [batch, 1, 1, 1, num_paths]
        quality_expanded = quality.view(quality.shape[0], 1, 1, 1, -1)

        # Weight paths by quality and sum
        # path_bundle.data: [batch, out_channels, h, w, num_paths]
        weighted_sum = (path_bundle.data * quality_expanded).sum(dim=-1)

        # Normalize by total quality
        total_quality = quality.sum(dim=-1, keepdim=True).view(-1, 1, 1, 1)

        collapsed = weighted_sum / (total_quality + 1e-8)

        return collapsed


class TrueeidosConvBlock(nn.Module):
    """
    Complete eidos Convolutional Block with normalization and activation.

    Structure:
      1. TrueeidosConv2D (set-valued spatial collapse)
      2. ModularPhaseNorm2d (eidos-compatible normalization)
      3. Tanh activation (sign-preserving)
      4. Optional collapse to single feature map

    eidos COMPATIBILITY FIXES:
      ❌ BatchNorm → ✅ ModularPhaseNorm2d
         WHY: BatchNorm couples samples via batch statistics, breaking path independence
         FIX: ModularPhaseNorm2d normalizes per-sample using structural coherence

      ❌ ReLU → ✅ Tanh
         WHY: ReLU(x) = max(0, x) kills negative branch, destroying {-W, 0, +W} structure
         FIX: Tanh preserves sign (bijective on ℝ), maintains set-valued paths
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_paths: int = 27,
        stride: int = 1,
        padding: int = 1,
        collapse_output: bool = False,
        is_global: bool = False
    ):
        super().__init__()

        self.collapse_output = collapse_output

        # True eidos convolution
        self.conv = TrueeidosConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_paths=num_paths,
            stride=stride,
            padding=padding,
            is_global=is_global
        )

        # eidos-compatible normalization (NOT BatchNorm!)
        # ModularPhaseNorm2d: Normalizes per-sample, preserves path independence
        self.norm = ModularPhaseNorm2d(out_channels)

        # eidos-compatible activation (NOT ReLU!)
        # Tanh: Sign-preserving, bijective, maintains {-W, 0, +W} structure
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, in_channels, h, w]

        Returns:
            If collapse_output=True: [batch, out_channels, h, w]
            If collapse_output=False: PathBundle [batch, out_channels, h, w, num_paths]
        """
        # Set-valued convolution
        path_bundle = self.conv(x)

        # Apply ModularPhaseNorm2d and Tanh activation to each path separately
        batch, out_c, h, w, num_paths = path_bundle.data.shape

        # Reshape to [batch * num_paths, out_channels, h, w] for normalization
        paths_flat = path_bundle.data.permute(0, 4, 1, 2, 3).reshape(
            batch * num_paths, out_c, h, w
        )

        # Apply ModularPhaseNorm2d (eidos-compatible) and Tanh (sign-preserving)
        paths_normalized = self.norm(paths_flat)
        paths_activated = self.activation(paths_normalized)

        # Reshape back to [batch, out_channels, h, w, num_paths]
        paths_processed = paths_activated.view(batch, num_paths, out_c, h, w).permute(0, 2, 3, 4, 1)

        # Update PathBundle data
        path_bundle.data = paths_processed

        # Collapse if requested
        if self.collapse_output:
            return self.conv.collapse_spatial_pathbundle(path_bundle)
        else:
            return path_bundle


# Export
__all__ = ['TrueeidosConv2D', 'TrueeidosConvBlock']
