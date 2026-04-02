"""
FORM-SPACE MAPPER

Maps sensory patterns → stable geometric positions in form-space.

Maps input patterns to stable geometric positions in a learned form-space
using multi-scale eidosTransform projections. Includes contrastive stability
loss to encourage consistent form-space clustering.

Architecture:
    1. Pattern → Geometric embedding (multi-scale)
    2. Contrastive stability (similar patterns → nearby positions)
    3. Cluster separation (different semantics → distant regions)
    4. Topology preservation (relationships maintained under transformation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from eidos_nn.layers.eidos_transform import eidosTransform, eidosSequential
from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm


class FormSpaceMapper(nn.Module):
    """
    Maps patterns to stable geometric positions in form-space.

    The form-space has hierarchical structure:
    1. Local neighborhoods (similar concepts)
    2. Regional clusters (semantic categories)
    3. Global topology (abstract relationships)

    Uses multi-scale eidos transforms to capture this hierarchy.
    """

    def __init__(
        self,
        pattern_dim: int,
        form_space_dim: int,
        num_scales: int = 3,
        stability_weight: float = 0.1
    ):
        """
        Args:
            pattern_dim: Input pattern dimensionality
            form_space_dim: Output form-space dimensionality
            num_scales: Number of hierarchical scales (local → regional → global)
            stability_weight: Weight for stability loss (contrastive)
        """
        super().__init__()
        self.pattern_dim = pattern_dim
        self.form_space_dim = form_space_dim
        self.num_scales = num_scales
        self.stability_weight = stability_weight

        # Multi-scale mapping (hierarchical form structure)
        self.scale_mappers = nn.ModuleList()
        current_dim = pattern_dim

        for scale_idx in range(num_scales):
            # Each scale refines the mapping
            scale_dim = pattern_dim + (form_space_dim - pattern_dim) * (scale_idx + 1) // num_scales

            mapper = eidosSequential(
                eidosTransform(current_dim, scale_dim, num_rotation_planes=4),
                ModularPhaseNorm(scale_dim, base=7)
            )
            self.scale_mappers.append(mapper)
            current_dim = scale_dim

        # Final projection to form-space
        self.final_projection = eidosSequential(
            eidosTransform(current_dim, form_space_dim, num_rotation_planes=4),
            ModularPhaseNorm(form_space_dim, base=7)
        )

        # Form stabilizer (encourages stable positions)
        self.form_stabilizer = eidosSequential(
            eidosTransform(form_space_dim, form_space_dim, num_rotation_planes=2),
            ModularPhaseNorm(form_space_dim, base=7)
        )

        print(f"[FormSpaceMapper] Initialized")
        print(f"  Pattern dim: {pattern_dim}")
        print(f"  Form-space dim: {form_space_dim}")
        print(f"  Scales: {num_scales} (local -> regional -> global)")
        print(f"  Stability weight: {stability_weight}")

    def forward(
        self,
        patterns: torch.Tensor,
        compute_stability_loss: bool = False,
        pattern_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Map patterns to form-space positions.

        Args:
            patterns: [batch, seq_len, pattern_dim] input patterns
            compute_stability_loss: Whether to compute contrastive stability loss
            pattern_pairs: Optional (positive_pairs, negative_pairs) for contrastive learning

        Returns:
            forms: [batch, seq_len, form_space_dim] positions in form-space
        """
        # Multi-scale hierarchical mapping
        x = patterns

        for scale_idx, mapper in enumerate(self.scale_mappers):
            x = mapper(x)
            # Residual connection for topology preservation
            if x.shape[-1] == patterns.shape[-1]:
                x = x + patterns

        # Final projection to form-space
        forms = self.final_projection(x)

        # Stabilization (smooth the positions)
        forms = self.form_stabilizer(forms)

        # Normalize to unit hypersphere (geometric constraint)
        forms = F.normalize(forms, dim=-1)

        return forms

    def compute_form_stability_loss(
        self,
        anchor_forms: torch.Tensor,
        positive_forms: torch.Tensor,
        negative_forms: torch.Tensor,
        margin: float = 0.5
    ) -> torch.Tensor:
        """
        Contrastive loss to encourage stable form positions.

        Similar patterns → nearby positions (small distance)
        Different patterns → distant positions (large distance)

        Args:
            anchor_forms: [batch, seq_len, form_dim] anchor positions
            positive_forms: [batch, seq_len, form_dim] similar patterns
            negative_forms: [batch, seq_len, form_dim] different patterns
            margin: Minimum separation between pos/neg

        Returns:
            loss: Contrastive stability loss
        """
        # Cosine similarity (on normalized forms)
        pos_similarity = F.cosine_similarity(
            anchor_forms.reshape(-1, self.form_space_dim),
            positive_forms.reshape(-1, self.form_space_dim),
            dim=-1
        )

        neg_similarity = F.cosine_similarity(
            anchor_forms.reshape(-1, self.form_space_dim),
            negative_forms.reshape(-1, self.form_space_dim),
            dim=-1
        )

        # Contrastive loss: pos should be close (sim → 1), neg should be far (sim → -1)
        # Distance = 1 - similarity
        pos_distance = 1.0 - pos_similarity
        neg_distance = 1.0 - neg_similarity

        # Triplet-style loss: encourage pos_dist < neg_dist - margin
        loss = F.relu(pos_distance - neg_distance + margin)

        return loss.mean()


class FormSpaceContrastiveMapper(FormSpaceMapper):
    """
    Form-space mapper with built-in contrastive learning.

    Automatically generates positive/negative pairs from batch
    to encourage stable form-space structure.
    """

    def __init__(
        self,
        pattern_dim: int,
        form_space_dim: int,
        num_scales: int = 3,
        stability_weight: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__(pattern_dim, form_space_dim, num_scales, stability_weight)
        self.temperature = temperature

        print(f"  Contrastive temperature: {temperature}")

    def compute_geometric_tension_loss(self, forms: torch.Tensor) -> torch.Tensor:
        """
        Enforce 'Geometry Basics' threading logic:
        B[i] - A[i] = Constant (4).
        
        We interpret this as: The vector difference between the 'Inner' (first half)
        and 'Outer' (second half) threads should be UNIVERSALLY CONSTANT across the batch.
        
        This aligns all concepts to the same underlying Lattice.
        """
        batch, dim = forms.shape
        half = dim // 2
        
        # Split into Threads
        thread_A = forms[:, :half] # Inner
        thread_B = forms[:, half:] # Outer
        
        # Calculate Gap
        gap = thread_B - thread_A
        
        # We want Gap to be constant across the batch.
        # Mean Gap represents the "Universal Structural Constant" (The "4").
        mean_gap = gap.mean(dim=0, keepdim=True)
        
        # Loss = Distance of each sample's gap from the Universal Gap
        # Minimize Variance.
        tension_loss = F.mse_loss(gap, mean_gap.expand_as(gap))
        
        return tension_loss

    def forward_with_contrastive(
        self,
        patterns: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with automatic contrastive loss computation + Geometric Tension.

        Args:
            patterns: [batch, seq_len, pattern_dim]
            labels: [batch] optional labels for supervised contrastive

        Returns:
            forms: [batch, seq_len, form_space_dim]
            loss: Combined stability loss
        """
        batch_size, seq_len, _ = patterns.shape

        # Map to form-space
        forms = self.forward(patterns)
        
        # 1. Geometric Tension (Internal Coherence)
        # Flatten sequence to treat every token as a geometric point
        flat_forms = forms.reshape(-1, self.form_space_dim)
        tension_loss = self.compute_geometric_tension_loss(flat_forms)

        # If no labels, return just tension loss (unsupervised structure)
        if labels is None:
            return forms, tension_loss * self.stability_weight

        # 2. Contrastive Loss (Semantic Clustering)
        # Pool across sequence for contrastive learning
        # (we want similar sentiments → similar forms at sentence level)
        pooled_forms = forms.mean(dim=1)  # [batch, form_dim]

        # Compute similarity matrix
        forms_norm = F.normalize(pooled_forms, dim=-1)
        similarity_matrix = torch.matmul(forms_norm, forms_norm.T) / self.temperature

        # Create label mask (positive pairs have same label)
        labels_expanded = labels.unsqueeze(1)  # [batch, 1]
        label_mask = (labels_expanded == labels_expanded.T).float()  # [batch, batch]

        # Mask out self-similarities
        mask_diagonal = torch.eye(batch_size, device=forms.device)
        label_mask = label_mask * (1 - mask_diagonal)

        # Supervised contrastive loss
        # For each anchor, pull positives close and push negatives away
        exp_sim = torch.exp(similarity_matrix)

        # Sum over all examples (denominator)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)

        # Log probability of positive pairs
        log_prob = similarity_matrix - torch.log(sum_exp_sim)

        # Mean over positive pairs
        num_positives = label_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1.0)  # Avoid division by zero

        contrastive_loss = -(label_mask * log_prob).sum(dim=1) / num_positives
        contrastive_loss = contrastive_loss.mean()

        # Combine losses (Semantic Clustering + Geometric Structure)
        total_loss = contrastive_loss + tension_loss
        return forms, total_loss * self.stability_weight
