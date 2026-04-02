"""
FRACTAL OPTIMIZER FOR SET-VALUED GRADIENTS

Implements multi-frequency learning rates derived from the Eidos axioms.
The gradient update is decomposed into three frequency bands, each scaled
by a different power of the batch-size ratio:

- 0.5 power (Coarse): Governs large-scale, stable anchor updates.
- 0.3 power (Triadic): Governs branching-scale updates relevant to
  the {-W, 0, +W} discrete potential structure.
- 0.2 power (Fine): Governs small-scale path diversity refinement.

The optimizer adapts the relative weighting of these bands based on
measured structural tension from the model's collapse layers.
"""

import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional
import numpy as np


class FractalOptimizer(Optimizer):
    """
    Optimizer using fractal learning rate decomposition.
    
    Directly implements parameter updates using fractal scaling laws.
    Removes dependence on classical Adam/SGD.
    
    Instead of a single LR, applies fractal scaling to gradients:
    - Coarse band (0.5): Stable anchor points
    - Triadic band (0.3): Set-valued branching
    - Fine band (0.2): Path diversity
    
    The effective update is a weighted composition of these fractal forces.
    """

    def __init__(self, params, base_lr: float = 0.0001, batch_scale: float = 1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Args:
            params: Model parameters
            base_lr: Base learning rate
            batch_scale: Ratio of current_batch_size / base_batch_size
            betas: Coefficients for computing running averages of gradient and its square (like Adam)
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        defaults = dict(base_lr=base_lr, batch_scale=batch_scale, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FractalOptimizer, self).__init__(params, defaults)

        self.base_lr = base_lr
        self.batch_scale = batch_scale
        self._swirl = "[FRACTAL]"

        # Fractal decomposition based on axioms
        self.lr_coarse = base_lr * (batch_scale ** 0.5)
        self.lr_triadic = base_lr * (batch_scale ** 0.3)
        self.lr_fine = base_lr * (batch_scale ** 0.2)


        # Track which frequency band is dominant (coarse, triadic, fine)
        # Normalized weights for the update composition
        self.frequency_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3] 

        print(f"\n{self._swirl} Fractal Optimizer initialized (Eidos Mode):")
        print(f"   Base LR: {base_lr:.6f} | Batch scale: {batch_scale}x")
        print(f"   Gradient Bands:")
        print(f"   +- Coarse (0.5):           lr = {self.lr_coarse:.6f}")
        print(f"   +- Triadic (0.3):          lr = {self.lr_triadic:.6f}")
        print(f"   +- Fine (0.2):             lr = {self.lr_fine:.6f}")

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        The update rule is a fractal composition of gradients scaled by 
        different frequency bands.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FractalOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Base step direction (Adam-like)
                step_size = math.sqrt(bias_correction2) / bias_correction1
                base_update = exp_avg / denom
                
                # --- FRACTAL COMPOSITION ---
                # Instead of a single LR, we compose the update from frequency bands
                
                # 1. Coarse Update (High stability, exploration)
                update_coarse = base_update * self.lr_coarse * self.frequency_weights[0]
                
                # 2. Triadic Update (Branching selection)
                # We can add a small perturbation/noise here if we wanted to simulate branching,
                # but for now we keep it as a distinct scale.
                update_triadic = base_update * self.lr_triadic * self.frequency_weights[1]
                
                # 3. Fine Update (Detail refinement)
                update_fine = base_update * self.lr_fine * self.frequency_weights[2]
                
                # Total Fractal Update
                total_update = (update_coarse + update_triadic + update_fine) * step_size
                
                p.data.add_(total_update, alpha=-1)

        return loss

    def measure_frequency_dominance(self) -> Dict[str, float]:
        # Placeholder - hard to measure dominance when updates are composed
        # Could return the current weights
        return {
            'coarse': self.frequency_weights[0],
            'triadic': self.frequency_weights[1],
            'fine': self.frequency_weights[2]
        }

    def adapt_frequencies(self, structural_tension: float):
        """
        Adjust frequency weights based on structural tension.
        """
        if structural_tension < 5e-8:
            self.frequency_weights = [0.6, 0.3, 0.1] # Emphasis on Coarse
        elif structural_tension > 2.5e-7:
            self.frequency_weights = [0.2, 0.6, 0.2] # Emphasis on Triadic
        else:
            self.frequency_weights = [0.33, 0.33, 0.33] # Balanced
    
    def adapt_from_loss(self, loss: float, loss_history: list = None):
        """
        Adjust frequency weights based on LOSS value directly.
        
        This is more holistic than tension-based adaptation because loss
        directly measures the training objective.
        
        Uses ceiling law concept from seven_frameworks:
        - High loss = need exploration (Triadic emphasis)
        - Medium loss = balanced exploration + refinement
        - Low loss = fine-tuning mode (Fine emphasis)
        
        Args:
            loss: Current batch loss value
            loss_history: Optional list of recent losses for trend detection
        """
        # Thresholds based on typical cross-entropy loss ranges
        # For overfit goal: loss < 1.0 means near-memorization
        HIGH_LOSS_THRESHOLD = 100.0  # Exploration needed
        LOW_LOSS_THRESHOLD = 10.0    # Fine-tuning mode
        OVERFIT_THRESHOLD = 1.0      # Near-perfect memorization
        
        if loss > HIGH_LOSS_THRESHOLD:
            # High loss: need exploration, use Triadic branching
            self.frequency_weights = [0.2, 0.6, 0.2]  # Triadic emphasis
        elif loss < OVERFIT_THRESHOLD:
            # Near-overfit: ultra-fine tuning
            self.frequency_weights = [0.1, 0.2, 0.7]  # Fine emphasis
        elif loss < LOW_LOSS_THRESHOLD:
            # Low loss: fine-tuning mode
            self.frequency_weights = [0.2, 0.3, 0.5]  # Fine skew
        else:
            # Medium loss: balanced
            self.frequency_weights = [0.33, 0.33, 0.33]

import math

class FractalScheduler:
    """
    Learning rate scheduler for FractalOptimizer.
    Handles linear warmup and tension-adaptive frequency rebalancing.
    """

    def __init__(self, optimizer: FractalOptimizer, warmup_batches: int = 100):
        self.optimizer = optimizer
        self.warmup_batches = warmup_batches
        self.current_batch = 0
        self.initial_lrs = [optimizer.lr_coarse, optimizer.lr_triadic, optimizer.lr_fine]

    def step(self, structural_tension: Optional[float] = None):
        self.current_batch += 1

        # Warmup phase
        if self.current_batch < self.warmup_batches:
            warmup_factor = self.current_batch / self.warmup_batches
            
            # Scale internal LRs
            self.optimizer.lr_coarse = self.initial_lrs[0] * warmup_factor
            self.optimizer.lr_triadic = self.initial_lrs[1] * warmup_factor
            self.optimizer.lr_fine = self.initial_lrs[2] * warmup_factor
            
        elif structural_tension is not None:
            self.optimizer.adapt_frequencies(structural_tension)
            # Restore full LRs after warmup
            self.optimizer.lr_coarse = self.initial_lrs[0]
            self.optimizer.lr_triadic = self.initial_lrs[1]
            self.optimizer.lr_fine = self.initial_lrs[2]

    def get_last_lr(self):
        return [self.optimizer.lr_coarse, self.optimizer.lr_triadic, self.optimizer.lr_fine]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        return {
            'warmup_batches': self.warmup_batches,
            'current_batch': self.current_batch,
            'initial_lrs': self.initial_lrs
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.warmup_batches = state_dict['warmup_batches']
        self.current_batch = state_dict['current_batch']
        self.initial_lrs = state_dict['initial_lrs']