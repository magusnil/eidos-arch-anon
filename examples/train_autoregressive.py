"""
AUTOREGRESSIVE DIAGNOSTIC FOR EIDOS (MOBIUS / Eidos)

Goal: Train a small Eidos model on a tiny text dataset until it overfits.
      This validates the autoregressive capability and the measurement-driven logic.

Author: Anonymous
Framework: Eidos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os
import random
import json
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add fprod_implementation to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eidos_nn.layers.eidos_transform import eidosTransform
from eidos_nn.layers.neighbor_mixer import eidosNeighborMixer
from eidos_nn.models.mobius_collapse_layer import MobiusCollapseLayer
from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm
from eidos_nn.optim.fractal_optimizer import FractalOptimizer, FractalScheduler
from eidos_nn.layers.form_space_mapper import FormSpaceContrastiveMapper
from eidos_nn.layers.set_valued_attention import SetValuedAttention
from eidos_nn.layers.ir_positional_encoding import IRHierarchicalPositionalEncoding

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
    def encode(self, s):
        return [self.char_to_idx[c] for c in s]
    
    def decode(self, ids):
        return "".join([self.idx_to_char[i] for i in ids])

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        # Create number mask for x
        # Check if each token ID corresponds to a digit
        mask = []
        for token_id in chunk[:-1]:
            char = self.tokenizer.idx_to_char[token_id]
            mask.append(char.isdigit())
        
        number_mask = torch.tensor(mask, dtype=torch.bool)
        
        return x, y, number_mask

class EidosAutoregressiveModel(nn.Module):
    """
    Autoregressive Eidos Model using Mobius Collapse.
    
    Implements full Mobius topology with:
    - Vertical coherence: collapsed_estimate passed between layers
    - Temporal coherence: collapsed_estimate carried between generation steps
    - Nonlinear stabilization: inter_step_twist applies R5/R6 geometric rotation
    """
    def __init__(self, vocab_size, d_model=256, num_layers=4, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = IRHierarchicalPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_regimes=5,
            use_rope=False  # Eidos: No RoPE
        )
        
        # Form-Space Mapper for initial stabilization
        self.form_mapper = FormSpaceContrastiveMapper(
            pattern_dim=d_model,
            form_space_dim=d_model,
            num_scales=2
        )
        
        # The Stack (Causal)
        # Causal Attention + Mobius Collapse stack
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': SetValuedAttention(d_model, num_heads=4),
                'mobius': MobiusCollapseLayer(d_model, num_paths=9, causal=True, num_twists=3)
            })
            for _ in range(num_layers)
        ])
        
        # Inter-step twist for temporal coherence (nonlinear stabilization)
        # This rotates the collapsed_estimate between generation steps
        self.inter_step_twist = eidosTransform(d_model, d_model, num_rotation_planes=2)
        
        self.norm_final = ModularPhaseNorm(d_model, base=7)
        # Prediction head: eidosTransform (Exp1 proved head is NOT the issue)
        self.head = eidosTransform(d_model, vocab_size, num_rotation_planes=2)
        
    def forward(self, x, number_mask=None, initial_collapsed=None):
        """
        Forward pass with Mobius coherence.
        """
        batch, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x_emb = self.token_emb(x)
        x = x_emb + self.pos_emb(pos)
        # Matrix size: [seq_len, seq_len]
        # visible if j <= i (lower triangular is 1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        # shape: [1, 1, seq_len, seq_len]
        
        # Mobius Propagation with VERTICAL COHERENCE
        # collapsed_estimate flows from layer to layer
        collapsed_estimate = initial_collapsed  # Start with temporal memory (or None)
        
        for layer in self.layers:
            # A. Global Field Interaction (Attention)
            # SetValuedAttention expects (q, k, v, mask)
            # Self-attention: q=k=v=x
            x_attn, _ = layer['attn'](x, x, x, mask=causal_mask)
            
            # Residual connection is NOT in SetValuedAttention, so we add it here
            x = x + x_attn
            
            # B. Local Geometric Collapse (Mobius)
            x, new_collapsed = layer['mobius'](x, initial_collapsed=collapsed_estimate)
            
            # Update collapsed estimate for next layer/step
            collapsed_estimate = new_collapsed
            
            # CEILING DAMPING (Seven Frameworks, Eq. 1)
            # k = ⌈M/N⌉ where M = magnitude, N = IR (information regime probable)
            # For training: simple damping (best loss: 286 vs 587 with carry)
            # For generation with stable models: enable use_r3_carry=True
            if self.training and collapsed_estimate is not None:
                IR_k = 5.0  # Information Regime probable (max magnitude per layer)
                use_r3_carry = False  # Enable for stable models during generation
                
                # Compute magnitude per-position (scale-aware via mean, not lossy norm)
                magnitude = collapsed_estimate.abs().mean(dim=-1, keepdim=True)
                
                # Ceiling law: how many quanta does this magnitude span?
                k = torch.ceil(magnitude / IR_k)
                k = torch.clamp(k, min=1.0)
                
                # Only damp if we exceed 1 probable (crossing regime boundary)
                if (k > 1).any():
                    # R6 SCALE: Project to single-probable regime
                    scale = IR_k / (magnitude + 1e-8)
                    scale = torch.clamp(scale, max=1.0)
                    collapsed_estimate = collapsed_estimate * scale
                    
                    # R3* CARRY: Optional overshoot preservation (for stable models)
                    if use_r3_carry:
                        overshoot = (k * IR_k) - magnitude
                        overshoot_normalized = overshoot / (k * IR_k + 1e-8)
                        phase_carry = overshoot_normalized * 0.1
                        direction = collapsed_estimate / (collapsed_estimate.abs().mean(dim=-1, keepdim=True) + 1e-8)
                        collapsed_estimate = collapsed_estimate + direction * phase_carry
            
        x = self.norm_final(x)
        
        # Re-inject positional encoding after Mobius collapse
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_emb(positions)  # Re-add position info before head
        
        logits = self.head(x)
        
        return logits, collapsed_estimate, None, x_emb
    
    def twist_for_next_step(self, collapsed_estimate, max_norm=10.0):
        """
        Apply nonlinear geometric rotation before handing off to next generation step.
        This prevents linear error accumulation while preserving phase coherence.
        
        We only carry the LAST position's collapsed state (the accumulated phase).
        This will be expanded to match the new sequence length when used.
        
        TENSION DAMPING (R6 Scale Axiom):
        If the memory norm exceeds max_norm, we apply R6 scaling to project
        back onto the stable manifold. This prevents runaway geometric accumulation.
        """
        if collapsed_estimate is None:
            return None
        # Take only the last position: [batch, seq, dim] -> [batch, 1, dim]
        last_state = collapsed_estimate[:, -1:, :]
        
        # Apply R5 rotation (the inter_step_twist)
        twisted = self.inter_step_twist(last_state)
        
        # R6 SCALE AXIOM: Tension-based damping
        # If ||twisted|| > max_norm, scale back to max_norm while preserving direction
        norm = torch.norm(twisted, dim=-1, keepdim=True)
        scale_factor = torch.clamp(max_norm / (norm + 1e-8), max=1.0)  # Only scale down, never up
        dampened = twisted * scale_factor
        
        return dampened
    
    def expand_memory_to_seq(self, collapsed_memory, seq_len):
        """
        Expand the carried memory [batch, 1, dim] to match sequence length.
        The single phase state is broadcast to all positions.
        """
        if collapsed_memory is None:
            return None
        # Expand: [batch, 1, dim] -> [batch, seq_len, dim]
        return collapsed_memory.expand(-1, seq_len, -1)

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device='cpu'):
    """
    Autoregressive generation with Mobius temporal coherence.
    
    The collapsed_estimate (last position's phase) is carried between generation steps,
    passing through inter_step_twist for nonlinear stabilization, then expanded
    to match the new sequence length.
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Initialize temporal memory (None = cold start)
    collapsed_memory = None  # Shape: [batch, 1, d_model] after twist
    
    for step in range(max_new_tokens):
        # We only take the last 'max_seq_len' tokens
        input_ids_chunk = input_ids[:, -128:]
        seq_len = input_ids_chunk.shape[1]
        
        # Expand memory to match current sequence length
        expanded_memory = model.expand_memory_to_seq(collapsed_memory, seq_len)
        
        with torch.no_grad():
            # Forward pass WITH temporal memory (expanded to match seq_len)
            logits, collapsed_estimate, _, _ = model(input_ids_chunk, initial_collapsed=expanded_memory)
            
            # Apply nonlinear twist and keep only last position for next step
            collapsed_memory = model.twist_for_next_step(collapsed_estimate)
            
            # Take last token's logits
            logits = logits[:, -1, :] / temperature
            
            # === R7/R10 SET-VALUED ERROR CORRECTION ===
            # R10 (Reset): If logits contain NaN/Inf, reset to {0, safe_value}
            # We interpret this as: replace bad values with the mean of good values
            nan_mask = ~torch.isfinite(logits)
            if nan_mask.any():
                safe_mean = logits[~nan_mask].mean() if (~nan_mask).any() else 0.0
                logits = torch.where(nan_mask, safe_mean, logits)
            
            # R7 (Oppositional): For numerical stability, clamp extreme values
            # †a† = -|a| principle: reflect extreme values toward the safe manifold
            logits = torch.clamp(logits, min=-100.0, max=100.0)
            
            # Standard softmax and sampling
            probs = F.softmax(logits, dim=-1)
            
            # Final safety: if probs still have issues, use uniform distribution
            if not torch.isfinite(probs).all() or (probs < 0).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            
    return tokenizer.decode(input_ids[0].cpu().numpy())

def train_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tiny dataset
    data_path = Path(__file__).parent / "eidos_diagnostic_text.txt"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return
        
    text = data_path.read_text()
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    seq_len = 32
    dataset = TextDataset(text, tokenizer, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = EidosAutoregressiveModel(tokenizer.vocab_size, d_model=128, num_layers=8).to(device)
    optimizer = FractalOptimizer(model.parameters(), base_lr=0.005)
    scheduler = FractalScheduler(optimizer)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 100 
    
    # === LOGGING SETUP ===
    run_dir = Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # File logger
    log_file = run_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Model: d_model=128, num_layers=8")
    
    # Metrics log (JSON for analysis)
    metrics_file = run_dir / "metrics.jsonl"
    best_loss = float('inf')
    
    print("\nStarting diagnostic training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for i, (x, y, mask) in enumerate(pbar):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            


            optimizer.zero_grad()
            
            # Training: we don't need temporal memory between batches
            # Training: we don't need temporal memory between batches
            logits, _, logic_target, x_emb = model(x, number_mask=mask, initial_collapsed=None)
            
            # 1. Standard Next-Token Prediction Loss
            next_token_loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            
            loss = next_token_loss
            
            if i % 10 == 0:
                 print(f"Step {i} Loss: {loss.item():.4f}")
            
            loss.backward()
            
            # R6 GRADIENT CLIPPING: Prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # --- GENERATION CHECK ---
            if i > 0 and i % 50 == 0:
                print(f"\n[Step {i}] Generating sample...")
                model.eval()
                with torch.no_grad():
                    # Seed with period (common starter)
                    # The generate function expects a string prompt, not token IDs.
                    # The `layout, _ = model(start_tokens)` line is not needed here
                    # as `generate` handles the initial forward pass.
                    
                    sample_text = generate(model, tokenizer, ".", max_new_tokens=100, temperature=0.8, device=device)
                    print(f"Sample: {repr(sample_text)}")
                model.train()
            
            # Simple tension check (kept for logging)
            all_grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            tension = torch.cat(all_grads).var().item() if all_grads else 0
            
            # Use LOSS-BASED adaptation instead of tension
            # This directly measures training progress
            optimizer.adapt_from_loss(loss.item())
            optimizer.step()
            scheduler.step(structural_tension=tension)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        if epoch % 1 == 0 or epoch == 1:  # Save metrics every epoch for live visualization
            avg_loss = total_loss/len(loader)
            
            # Generate sample
            gen_text = generate(model, tokenizer, "The cat ", max_new_tokens=30, device=device)
            
            # Log to console and file
            logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Tension: {tension:.2e}")
            logger.info(f"  Sample: {repr(gen_text)}")
            
            # Save metrics to JSONL
            metrics = {
                'epoch': epoch,
                'avg_loss': avg_loss,
                'tension': tension,
                'sample': gen_text,
                'frequency_weights': optimizer.frequency_weights
            }
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Save checkpoint on best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = run_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'sample': gen_text
                }, checkpoint_path)
                logger.info(f"  [CHECKPOINT] New best loss! Saved to {checkpoint_path.name}")
        
        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(loader)
            }, checkpoint_path)
            logger.info(f"  [CHECKPOINT] Periodic save: {checkpoint_path.name}")

    print("\nDiagnostic complete.")
    # Final overfitting check
    gen_text = generate(model, tokenizer, "The cat ", max_new_tokens=50, device=device)
    print(f"\nFinal Overfit Generation:\n{gen_text}")

if __name__ == "__main__":
    train_diagnostic()
