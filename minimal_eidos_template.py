"""
Minimal eidos Template - v4.1 (Universal)

A generic starting point for ANY task (Classification, Regression, or Sequence Modeling).

Core Components:
1. eidos Block: The fundamental atom (Pre-Norm -> Set-Attn -> FFN).
2. Fractal Optimizer: Required for multi-frequency learning.
3a. Hierarchical Scorer: Prevents path explosion.
OR
3b. Probable Collapse Layer: Prevents path explosion. (this is a faster scorer + learner, trains faster overall.)
3c. MobiusCollapseLayer is needed for joint classification + regression tasks.

Usage:
- Copy 'eidosClassifier' for tabular/image tasks.
- Copy 'eidosSequenceModel' for text/time-series tasks.
"""

import argparse
import random
import torch
import torch.nn as nn
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# --- CORE IMPORTS WHICH ARE TASK DEPENDENT ---
from eidos_nn.layers.set_valued_attention import SetValuedAttention
from eidos_nn.layers.true_eidos_ffn import TrueeidosFFN as TrueEidosFFN
from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm
from eidos_nn.optim.fractal_optimizer import FractalOptimizer, FractalScheduler
# PathBundle is used internally by layers; not needed for direct model building.
from eidos_nn.layers.eidos_transform import eidosTransform, eidosSequential  # in general
from eidos_nn.layers.convolution import ModularPhaseNorm2d  # For image tasks
from eidos_nn.utils.measure_structural_tension import measure_path_tension # to measure tension
from eidos_nn.utils.logger import eidosLogger

# --- LOGIC ENGINE (Optional) // EXPERIMENTAL AND UNUSED ---
try:
    LOGIC_AVAILABLE = False
    LOGIC_AVAILABLE = True
except ImportError:
    LOGIC_AVAILABLE = False

# -- For Text examples
# from eidos_nn.layers.eidos_transform import eidosTransform, eidosSequential  # in general
# from eidos_nn.optim.fractal_optimizer import FractalOptimizer, FractalScheduler
# from eidos_nn.utils.measure_structural_tension import measure_path_tension # to measure tension
# from eidos_nn.utils.modular_phase_norm import ModularPhaseNorm




class MinimalEidosBlock(nn.Module):
    """
    The Universal Atom of Eidos Networks.
    Stabilized with Pre-Norm and Residuals.
    """
    def __init__(self, d_model, num_heads=4, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.norm1 = ModularPhaseNorm(d_model, base=7)
        self.attn = SetValuedAttention(d_model, num_heads, dropout)
        
        self.norm2 = ModularPhaseNorm(d_model, base=7)
        self.ffn = TrueEidosFFN(d_model, d_model * expansion_factor, num_paths=9, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class EidosClassifier(nn.Module):
    """
    Generic Eidos Model for Classification / Regression.
    Input: [Batch, Features] -> Output: [Batch, Classes]
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        # Geometric Embedding (Project input to hidden dimension)
        self.embedding = eidosTransform(input_dim, hidden_dim, num_rotation_planes=2)
        
        # Deep Reasoning Blocks
        self.layers = nn.ModuleList([
            MinimalEidosBlock(hidden_dim, num_heads=4) for _ in range(num_layers)
        ])
        
        # Classifier Head (Global Pool -> Transform)
        self.norm_final = ModularPhaseNorm(hidden_dim, base=7)
        self.head = eidosTransform(hidden_dim, output_dim, num_rotation_planes=2)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        return self.head(x)

class EidosSequenceModel(nn.Module):
    """
    Generic Eidos Model for Sequence Tasks (Text/Time-Series).
    Input: [Batch, Seq_Len, Input_Dim] -> Output: [Batch, Seq_Len, Output_Dim]
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.embedding = eidosTransform(input_dim, hidden_dim, num_rotation_planes=2)
        
        self.layers = nn.ModuleList([
            MinimalEidosBlock(hidden_dim, num_heads=4) for _ in range(num_layers)
        ])
        
        self.norm_final = ModularPhaseNorm(hidden_dim, base=7)
        self.head = eidosTransform(hidden_dim, output_dim, num_rotation_planes=2)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm_final(x)
        return self.head(x)


def set_reproducibility(seed: int):
    """
    Set the obvious RNG hooks for a run.
    Same numeric seeds are useful, but they are not a full cross-platform
    reproducibility guarantee once backend and kernel differences are involved.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_dummy_data(mode='classification'):
    if mode == 'classification':
        x = torch.randn(32, 16)
        y = torch.randint(0, 2, (32,))
        return x, y

    x = torch.randn(16, 8, 16)
    y = torch.randn(16, 8, 16)
    return x, y

def train_template(mode='classification', seed=42, log_dir='logs/template'):
    print(f"\n{'='*40}\nTraining Template: {mode.upper()}\n{'='*40}")
    set_reproducibility(seed)
    
    # 1. Setup Data & Model
    X, y = get_dummy_data(mode)
    
    if mode == 'classification':
        model = EidosClassifier(16, 2)
        criterion = nn.CrossEntropyLoss()
    else:
        model = EidosSequenceModel(16, 16) # Seq-to-Seq
        criterion = nn.MSELoss() # Or CrossEntropy for tokens
        
    # 2. Setup Fractal Optimizer (CRITICAL)
    optimizer = FractalOptimizer(model.parameters(), base_lr=0.001)
    scheduler = FractalScheduler(optimizer, warmup_batches=5)
    logger = eidosLogger(
        run_name=f"minimal_{mode}",
        config={
            "model_name": type(model).__name__,
            "dataset": f"dummy_{mode}",
            "optimizer": "FractalOptimizer",
            "lr": 0.001,
            "num_epochs": 1,
            "seed": seed,
            "params": sum(p.numel() for p in model.parameters()),
        },
        log_dir=log_dir,
    )
    
    # 3. Loop
    model.train()
    for step in range(10): # Dummy steps
        optimizer.zero_grad()
        out = model(X)
        
        # Loss Calc
        if mode == 'classification': loss = criterion(out, y)
        else: loss = criterion(out, y) # Dummy target match
            
        loss.backward()
        
        # Tension & Fractal Step
        tension = 0.0 # Implement measure_tension() here in real code
        optimizer.adapt_frequencies(tension)
        optimizer.step()
        scheduler.step()
        
        print(f"Step {step}: Loss {loss.item():.4f}")
        logger.log_progress(1, step + 1, 10, loss.item(), 0.0)
        
    logger.log_epoch({"epoch": 1, "train_loss": float(loss.item())})
    logger.finalize({"notes": "Minimal template smoke run complete."})

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['classification', 'sequence', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=str, default='logs/template')
    args = parser.parse_args()

    if args.mode in ('classification', 'both'):
        train_template('classification', seed=args.seed, log_dir=args.log_dir)
    if args.mode in ('sequence', 'both'):
        train_template('sequence', seed=args.seed, log_dir=args.log_dir)

    if LOGIC_AVAILABLE:
        print("\nNote: EidosLogicEngine is available for enhanced reasoning tasks.")
