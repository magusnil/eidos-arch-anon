"""
SET-VALUED ATTENTION (eidos Compliant)
Dual-Mode: Supports both legacy (query, key, value, mask) and new (x, use_cache) APIs.

Key Features:
1. W_q, W_k, W_v are eidosTransforms (Geometric Rotations) instead of Linear.
2. Output projection is eidosTransform.
3. Normalization uses ModularPhaseNorm (handled in Block usually).

Uses eidosTransform for Q/K/V projections (geometric rotations) 
instead of standard nn.Linear projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Handle imports assuming we are running from root or within module
try:
    from .eidos_transform import eidosTransform
    from ..utils.modular_phase_norm import ModularPhaseNorm
except ImportError:
    # Use absolute import as fallback
    from eidos_nn.layers.eidos_transform import eidosTransform
    from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm


class SetValuedAttention(nn.Module):
    """
    Geometric Set-Valued Attention (eidos Compliant).
    
    Supports TWO modes:
    1. Legacy: forward(query, key, value, mask) - for EidosAutoregressiveModel
    2. Modern: forward(x, use_cache, past_key_value) - for eidos_LLM (27-path vectorized)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # Geometric Projections (No nn.Linear!)
        self.q_proj = eidosTransform(d_model, d_model, num_rotation_planes=4)
        self.k_proj = eidosTransform(d_model, d_model, num_rotation_planes=4)
        self.v_proj = eidosTransform(d_model, d_model, num_rotation_planes=4)
        self.out_proj = eidosTransform(d_model, d_model, num_rotation_planes=4)
        
        # eidos Norm for stability
        self.output_norm = ModularPhaseNorm(d_model, base=7)
        self.input_norm = ModularPhaseNorm(d_model, base=7)
        self.q_norm = ModularPhaseNorm(d_model, base=7)
        self.k_norm = ModularPhaseNorm(d_model, base=7)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key=None, value=None, mask=None):
        """
        Dual-Mode Forward:
        
        Mode 1 (Legacy): forward(query, key, value, mask)
            - query, key, value: [batch, seq_len, d_model]
            - mask: [1, 1, seq_len, seq_len] or None
            
        Mode 2 (Self-Attention): forward(x) where x is query=key=value
            - x: [batch, seq_len, d_model]
        
        Returns: (output, attention_weights) tuple
        """
        # Handle self-attention case (only query provided)
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, d_model = query.shape
        
        # Pre-Norm: Normalize inputs to geometric regime
        query = self.input_norm(query)
        key = self.input_norm(key)
        value = self.input_norm(value)
        
        # 1. Geometric Projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Normalize Q/K to prevent dot product explosion from eidos scaling
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # 2. Reshape for Heads [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # --- FLOAT64 PRECISION BLOCK (prevents softmax overflow) ---
        q_64 = Q.double()
        k_64 = K.double()
        v_64 = V.double()
        
        # 3. Scaled Dot-Product Attention [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q_64, k_64.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # mask shape: [batch, 1, 1, seq_len] or [1, 1, seq_len, seq_len]
            mask_val = -1e9
            scores = scores.masked_fill(mask == 0, mask_val)
            
        # Softmax in double precision
        attn = torch.softmax(scores, dim=-1)
        
        # 4. Aggregation [batch, num_heads, seq_len, d_k]
        context = torch.matmul(attn, v_64)
        
        # Cast back to original dtype
        context = context.to(query.dtype)
        attn = attn.to(query.dtype)
        
        # 5. Re-assemble [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 6. Final Geometric Projection
        output = self.out_proj(context)
        
        # 7. eidos Normalization (Critical for stability)
        output = self.output_norm(output)
        
        return output, attn

