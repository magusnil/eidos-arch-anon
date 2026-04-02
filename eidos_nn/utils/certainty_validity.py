"""
Certainty-Validity Evaluation Framework

Traditional metrics (accuracy, AUROC, precision, recall) treat all errors equally,
ignoring whether the model committed confidently or expressed appropriate uncertainty.
This is inadequate for discrete selection systems.

The Certainty-Validity Matrix distinguishes four cases:
    - Confident-Correct (CC): The ideal outcome
    - Confident-Incorrect (CI): Overconfident error (failure mode)
    - Uncertain-Correct (UC): Lucky guess despite uncertainty
    - Uncertain-Incorrect (UI): Appropriate epistemic uncertainty

Crucially, UI represents principled behavior—the model correctly identified an
ambiguous case even if its forced prediction was incorrect.

Derived metrics:
    - CommitAcc: Reliability when the model commits (CC / (CC + CI))
    - AppropUncert: Fraction of errors flagged as uncertain (UI / (CI + UI))
    - Coverage: Fraction of samples the model commits to ((CC + CI) / total)
    - CVS: Certainty-Validity Score (CommitAcc × AppropUncert)

Example interpretation:
    A model with 83% accuracy and CVS=0.72 (CommitAcc=93%, AppropUncert=77%)
    is fundamentally different from 83% accuracy with CVS=0.18. The former
    is reliable when certain and knows what it doesn't know; the latter is
    overconfident on ambiguous samples.

Applications:

    Machine Learning / Safety-Critical Systems:
        For medical diagnosis, autonomous vehicles, and content moderation,
        CVS > raw accuracy. You want a model that is reliable when it commits
        and appropriately uncertain otherwise.
    
    Game Design / Playtesting:
        The CVS framework maps directly to player experience analysis:
        
        - CC (Confident-Correct): Genre enthusiasts who confidently expect to
          enjoy a game and do. These are your core audience—marketing matched
          their expectations perfectly.
        
        - CI (Confident-Incorrect): The critical failure mode. Players who
          confidently expected one experience but got another. This indicates:
          * Marketing/messaging doesn't match actual game design
          * Onboarding fails to set correct expectations
          * Genre conventions are violated without warning
          High CI = "expectation mismatch" = negative reviews, refunds
        
        - UC (Uncertain-Correct): Non-gamers or genre-outsiders who try
          something uncertain and happen to enjoy it. Discovery success—
          these can become new fans.
        
        - UI (Uncertain-Incorrect): Players trying new genres or games that
          do something different. They were appropriately uncertain and their
          uncertainty was validated. This is NOT a failure—it's honest
          expectation-setting for an experimental experience.
        
        Key insight: CI is the playtesting metric for "how well do you
        understand the onboarding of a game or a specific ask involving a game."
        A game with low CI has honest marketing and clear communication.
        A game with high CI misleads players about what they're getting.

"""

import torch
import numpy as np
from typing import Dict, Union, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CertaintyValidityResult:
    """Container for Certainty-Validity evaluation results."""
    CC: int  # Confident-Correct
    CI: int  # Confident-Incorrect
    UC: int  # Uncertain-Correct
    UI: int  # Uncertain-Incorrect
    CommitAcc: float  # Reliability when certain
    AppropUncert: float  # Knows what it doesn't know
    Coverage: float  # Fraction committed to
    CVS: float  # Composite score
    accuracy: float  # Standard accuracy for comparison
    threshold: float  # Certainty threshold used
    
    def __str__(self) -> str:
        return (
            f"Certainty-Validity Matrix:\n"
            f"  CC={self.CC:5d}  CI={self.CI:5d}  (certain)\n"
            f"  UC={self.UC:5d}  UI={self.UI:5d}  (uncertain)\n"
            f"Metrics:\n"
            f"  CommitAcc:    {self.CommitAcc:.4f} (reliability when certain)\n"
            f"  AppropUncert: {self.AppropUncert:.4f} (knows what it doesn't know)\n"
            f"  Coverage:     {self.Coverage:.4f} (fraction committed)\n"
            f"  CVS:          {self.CVS:.4f} (composite score)\n"
            f"  Accuracy:     {self.accuracy:.4f} (standard metric)\n"
            f"  Threshold:    {self.threshold:.4f}"
        )
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary for logging."""
        return {
            'CC': self.CC,
            'CI': self.CI,
            'UC': self.UC,
            'UI': self.UI,
            'CommitAcc': self.CommitAcc,
            'AppropUncert': self.AppropUncert,
            'Coverage': self.Coverage,
            'CVS': self.CVS,
            'accuracy': self.accuracy,
            'threshold': self.threshold
        }


def compute_cvs(
    predictions: Union[torch.Tensor, np.ndarray],
    certainty_scores: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> CertaintyValidityResult:
    """
    Compute the Certainty-Validity Score and related metrics.
    
    Args:
        predictions: Model predictions (class indices)
        certainty_scores: Certainty/confidence scores in [0, 1]
                         Can be softmax max, entropy-derived, or explicit certainty
        labels: Ground truth labels
        threshold: Certainty threshold for "high certainty" classification
                  Samples with certainty > threshold are considered "certain"
    
    Returns:
        CertaintyValidityResult with all metrics
    
    Example:
        >>> preds = torch.tensor([0, 1, 0, 1, 0])
        >>> certainty = torch.tensor([0.9, 0.6, 0.3, 0.4, 0.8])
        >>> labels = torch.tensor([0, 1, 1, 0, 0])
        >>> result = compute_cvs(preds, certainty, labels, threshold=0.5)
        >>> print(result.CVS)  # Composite score
    """
    # Convert to numpy for consistent handling
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(certainty_scores, torch.Tensor):
        certainty_scores = certainty_scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Flatten if needed
    predictions = predictions.flatten()
    certainty_scores = certainty_scores.flatten()
    labels = labels.flatten()
    
    # Classification
    certain = certainty_scores > threshold
    correct = predictions == labels
    
    # Compute the four quadrants
    CC = int(np.sum(certain & correct))
    CI = int(np.sum(certain & ~correct))
    UC = int(np.sum(~certain & correct))
    UI = int(np.sum(~certain & ~correct))
    
    # Derived metrics with safe division
    CommitAcc = CC / (CC + CI) if (CC + CI) > 0 else 0.0
    # Note: If no errors (CI + UI == 0), we define AppropUncert as 1.0 by convention.
    # This represents the ideal case where there are no errors to be uncertain about.
    # The metric becomes meaningful when there are actual errors to evaluate.
    AppropUncert = UI / (CI + UI) if (CI + UI) > 0 else 1.0
    Coverage = (CC + CI) / len(labels) if len(labels) > 0 else 0.0
    CVS = CommitAcc * AppropUncert
    accuracy = np.mean(correct)
    
    return CertaintyValidityResult(
        CC=CC, CI=CI, UC=UC, UI=UI,
        CommitAcc=CommitAcc,
        AppropUncert=AppropUncert,
        Coverage=Coverage,
        CVS=CVS,
        accuracy=float(accuracy),
        threshold=threshold
    )


def compute_cvs_from_logits(
    logits: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    certainty_method: str = 'softmax_max'
) -> CertaintyValidityResult:
    """
    Compute CVS directly from model logits.
    
    Args:
        logits: Raw model output (batch, num_classes)
        labels: Ground truth labels
        threshold: Certainty threshold
        certainty_method: How to derive certainty from logits
            - 'softmax_max': max(softmax(logits))
            - 'entropy': 1 - normalized_entropy(softmax(logits))
            - 'margin': difference between top two logits, normalized
    
    Returns:
        CertaintyValidityResult
    """
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Compute certainty
    probs = torch.softmax(logits, dim=-1)
    
    if certainty_method == 'softmax_max':
        certainty = probs.max(dim=-1).values
        
    elif certainty_method == 'entropy':
        # Lower entropy = higher certainty
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        max_entropy = np.log(logits.shape[-1])  # Uniform distribution
        certainty = 1.0 - (entropy / max_entropy)
        
    elif certainty_method == 'margin':
        # Larger margin between top two = higher certainty
        num_classes = logits.shape[-1]
        if num_classes < 2:
            raise ValueError("'margin' method requires at least 2 classes")
        top2 = probs.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        certainty = margin  # Already in [0, 1]
        
    else:
        raise ValueError(f"Unknown certainty_method: {certainty_method}")
    
    return compute_cvs(predictions, certainty, labels, threshold)


def find_optimal_threshold(
    predictions: Union[torch.Tensor, np.ndarray],
    certainty_scores: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, CertaintyValidityResult]:
    """
    Find the threshold that maximizes CVS.
    
    Args:
        predictions: Model predictions
        certainty_scores: Certainty scores
        labels: Ground truth labels
        thresholds: Thresholds to try (default: 0.1 to 0.9 in steps of 0.05)
    
    Returns:
        (optimal_threshold, CertaintyValidityResult at that threshold)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    best_cvs = -1
    best_threshold = 0.5
    best_result = None
    
    for thresh in thresholds:
        result = compute_cvs(predictions, certainty_scores, labels, thresh)
        if result.CVS > best_cvs:
            best_cvs = result.CVS
            best_threshold = thresh
            best_result = result
    
    return best_threshold, best_result


def compare_models(
    results: Dict[str, CertaintyValidityResult],
    print_comparison: bool = True
) -> str:
    """
    Compare CVS results across multiple models.
    
    Args:
        results: Dictionary mapping model names to CertaintyValidityResult
        print_comparison: Whether to print the comparison table
    
    Returns:
        Formatted comparison string
    """
    header = f"{'Model':<20} {'Accuracy':>10} {'CommitAcc':>10} {'AppropUncert':>12} {'CVS':>8} {'Coverage':>10}"
    divider = "-" * len(header)
    
    lines = [header, divider]
    
    for name, result in sorted(results.items(), key=lambda x: -x[1].CVS):
        line = (f"{name:<20} {result.accuracy:>10.4f} {result.CommitAcc:>10.4f} "
                f"{result.AppropUncert:>12.4f} {result.CVS:>8.4f} {result.Coverage:>10.4f}")
        lines.append(line)
    
    output = "\n".join(lines)
    
    if print_comparison:
        print(output)
    
    return output


class CertaintyValidityTracker:
    """
    Track CVS metrics during training.
    
    Usage:
        tracker = CertaintyValidityTracker()
        
        for batch in dataloader:
            outputs = model(inputs)
            # ... training step ...
            tracker.update(outputs, labels)
        
        result = tracker.compute()
        print(result)
    """
    
    def __init__(self, threshold: float = 0.5, certainty_method: str = 'softmax_max'):
        self.threshold = threshold
        self.certainty_method = certainty_method
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.all_predictions = []
        self.all_certainties = []
        self.all_labels = []
    
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Add batch predictions to tracker.
        
        Args:
            logits: Model output logits (batch, num_classes)
            labels: Ground truth labels
        """
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)
            
            if self.certainty_method == 'softmax_max':
                certainty = probs.max(dim=-1).values
            elif self.certainty_method == 'entropy':
                eps = 1e-8
                entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
                max_entropy = np.log(logits.shape[-1])
                certainty = 1.0 - (entropy / max_entropy)
            elif self.certainty_method == 'margin':
                num_classes = logits.shape[-1]
                if num_classes < 2:
                    raise ValueError("'margin' method requires at least 2 classes")
                top2 = probs.topk(2, dim=-1).values
                certainty = top2[:, 0] - top2[:, 1]
            else:
                raise ValueError(f"Unknown certainty_method: {self.certainty_method}")
            
            self.all_predictions.append(predictions.cpu())
            self.all_certainties.append(certainty.cpu())
            self.all_labels.append(labels.cpu())
    
    def compute(self) -> CertaintyValidityResult:
        """Compute CVS from accumulated predictions.
        
        Raises:
            ValueError: If no data has been accumulated via update()
        """
        if not self.all_predictions:
            raise ValueError("No data accumulated. Call update() with batch data first.")
        
        predictions = torch.cat(self.all_predictions)
        certainties = torch.cat(self.all_certainties)
        labels = torch.cat(self.all_labels)
        
        return compute_cvs(predictions, certainties, labels, self.threshold)


# Convenience function for quick evaluation
def evaluate_with_cvs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> CertaintyValidityResult:
    """
    Evaluate a model using CVS metrics.
    
    Args:
        model: PyTorch model
        dataloader: Data loader for evaluation
        device: Device to run on
        threshold: Certainty threshold
    
    Returns:
        CertaintyValidityResult
    """
    model.eval()
    tracker = CertaintyValidityTracker(threshold=threshold)
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels = batch[0], batch[1]
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            tracker.update(outputs, labels)
    
    return tracker.compute()


if __name__ == '__main__':
    # Example usage and test
    print("Certainty-Validity Framework Demo")
    print("=" * 50)
    
    # Simulate a discrete selection model (high CVS)
    np.random.seed(42)
    n_samples = 1000
    
    # Model A: Discrete selection (uncertain on ambiguous)
    labels_a = np.random.randint(0, 2, n_samples)
    # 85% correct with high certainty, 15% uncertain (some correct, some not)
    preds_a = labels_a.copy()
    certain_mask = np.random.random(n_samples) > 0.15  # 85% certain
    flip_mask = ~certain_mask & (np.random.random(n_samples) > 0.5)
    preds_a[flip_mask] = 1 - preds_a[flip_mask]
    certainty_a = np.where(certain_mask, 0.9 + np.random.random(n_samples) * 0.1, 0.3 + np.random.random(n_samples) * 0.2)
    
    result_a = compute_cvs(preds_a, certainty_a, labels_a, threshold=0.5)
    print("\nModel A (Discrete Selection - appropriate uncertainty):")
    print(result_a)
    
    # Model B: Overconfident baseline (low CVS)
    preds_b = labels_a.copy()
    # Same 85% correct, but always high certainty
    error_mask_b = np.random.random(n_samples) > 0.85
    preds_b[error_mask_b] = 1 - preds_b[error_mask_b]
    certainty_b = 0.8 + np.random.random(n_samples) * 0.2  # Always confident
    
    result_b = compute_cvs(preds_b, certainty_b, labels_a, threshold=0.5)
    print("\nModel B (Overconfident Baseline):")
    print(result_b)
    
    # Compare
    print("\n" + "=" * 50)
    print("Comparison:")
    compare_models({
        'Discrete Selection': result_a,
        'Overconfident Baseline': result_b
    })
