"""
STRUCTURAL TENSION MEASUREMENT

Measures path coordination tension during training. Tension is defined
as the variance across parallel path predictions: high tension indicates
paths are pulling in different directions (diversification), low tension
indicates alignment (convergence).

This metric can be used to detect phase transitions where the model
shifts from random exploration to structured specialization.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader


def measure_path_tension(model: nn.Module, batch_x: torch.Tensor,
                        device: torch.device) -> float:
    """
    Measure structural tension from path variance.

    High tension = paths pulling different directions
    Low tension = paths aligned

    Returns:
        tension: Path coordination tension metric
    """
    model.eval()

    with torch.no_grad():
        # Forward pass to get path outputs
        output = model(batch_x)

        # Get path-level predictions if available
        # For V3, we need to access intermediate path activations
        if hasattr(model, 'get_path_predictions'):
            path_preds = model.get_path_predictions(batch_x)
            # Tension = variance across paths
            tension = path_preds.var(dim=-1).mean().item()
        else:
            # Fallback: Use output variance as proxy
            # Lower bound on actual path tension
            tension = output.var().item()

    model.train()
    return tension


def measure_quality_score_variance(model: nn.Module, batch_x: torch.Tensor,
                                   device: torch.device) -> float:
    """
    Measure variance in path quality scores.

    During equilibrium: All paths have similar low scores
    During transition: Scores diverge (some paths dominate)
    After transition: Stable high variance (paths specialized)
    """
    model.eval()

    with torch.no_grad():
        # Access hierarchical scorer if available
        if hasattr(model, 'hierarchical_scorer'):
            # Get path quality scores
            # This requires forward pass through embedding
            embeddings = model.embedding(batch_x)

            if hasattr(embeddings, 'quality_scores'):
                scores = embeddings.quality_scores
                tension = scores.var(dim=-1).mean().item()
            else:
                tension = 0.0
        else:
            tension = 0.0

    model.train()
    return tension


def train_with_tension_tracking(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_file: Path
) -> Dict:
    """
    Training loop with structural tension tracking.

    Returns:
        metrics: Dict with loss, accuracy, and tension per batch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    batch_metrics = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Measure BEFORE update (current state tension)
        tension = measure_path_tension(model, batch_x, device)
        score_var = measure_quality_score_variance(model, batch_x, device)

        # Standard training step
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

        acc = 100. * correct / total

        # Log metrics
        batch_metrics.append({
            'batch': batch_idx,
            'loss': loss.item(),
            'acc': acc,
            'tension': tension,
            'score_variance': score_var,
        })

        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2f}%',
            'tension': f'{tension:.4f}'
        })

        # Save periodically
        if batch_idx % 50 == 0:
            with open(log_file, 'w') as f:
                json.dump(batch_metrics, f, indent=2)

    # Final save
    with open(log_file, 'w') as f:
        json.dump(batch_metrics, f, indent=2)

    return {
        'train_loss': total_loss / len(train_loader),
        'train_acc': 100. * correct / total,
        'batch_metrics': batch_metrics
    }


def analyze_tension_evolution(metrics: List[Dict], output_path: Path):
    """
    Analyze and visualize structural tension evolution.

    Detects:
    1. Equilibrium phase (low tension)
    2. Approach to IR_2 (rising tension)
    3. Phase transition (spike)
    4. Post-transition (stable high tension)
    """
    batches = [m['batch'] for m in metrics]
    tensions = [m['tension'] for m in metrics]
    accuracies = [m['acc'] for m in metrics]

    # Detect phase transition
    threshold = 1.8  # From theory
    transition_batches = [b for b, t in zip(batches, tensions) if t > threshold]

    if transition_batches:
        transition_point = min(transition_batches)
        transition_pct = 100 * transition_point / max(batches)
    else:
        transition_point = None
        transition_pct = None

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot tension
    ax1.plot(batches, tensions, label='Structural Tension', color='red', linewidth=2)
    ax1.axhline(threshold, color='orange', linestyle='--', label=f'IR_2 Threshold ({threshold})')

    if transition_point is not None:
        ax1.axvline(transition_point, color='green', linestyle='--',
                   label=f'Detected Transition (batch {transition_point}, {transition_pct:.1f}%)')

    ax1.set_ylabel('Structural Tension', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Structural Tension Evolution (IR_2 Boundary Crossing)', fontsize=14, fontweight='bold')

    # Plot accuracy
    ax2.plot(batches, accuracies, label='Training Accuracy', color='blue', linewidth=2)

    if transition_point is not None:
        ax2.axvline(transition_point, color='green', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Batch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Print analysis
    print("\n" + "=" * 70)
    print("STRUCTURAL TENSION ANALYSIS")
    print("=" * 70)

    print(f"\nTotal batches: {len(batches)}")
    print(f"Theory prediction: Transition at ~70% ({int(0.70 * len(batches))} batches)")

    if transition_point is not None:
        print(f"\n✅ Phase transition detected!")
        print(f"   Batch: {transition_point}")
        print(f"   Progress: {transition_pct:.1f}%")
        print(f"   Accuracy at transition: {accuracies[transition_point]:.2f}%")

        # Calculate accuracy jump
        if transition_point < len(batches) - 50:
            pre_acc = accuracies[max(0, transition_point - 10)]
            post_acc = accuracies[min(len(batches) - 1, transition_point + 50)]
            acc_jump = post_acc - pre_acc
            print(f"   Accuracy jump: +{acc_jump:.2f}%")
    else:
        print(f"\n⚠️  No phase transition detected (tension never exceeded {threshold})")
        print(f"   Max tension: {max(tensions):.4f}")
        print(f"   Model may need more batches to reach IR_2 boundary")

    # Tension statistics by phase
    third = len(batches) // 3
    print(f"\nTension evolution by phase:")
    print(f"  Early (0-33%):     Mean = {sum(tensions[:third])/third:.4f}")
    print(f"  Middle (33-66%):   Mean = {sum(tensions[third:2*third])/third:.4f}")
    print(f"  Late (66-100%):    Mean = {sum(tensions[2*third:])/len(tensions[2*third:]):.4f}")

    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Measure structural tension during training')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output-dir', type=str, default='tension_analysis')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Load data
    print("\nLoading IMDB dataset...")
    train_dataset = IMDBDataset(split='train', max_len=256, vocab_size=10000)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)

    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Expected IR_2 transition: ~{int(0.70 * len(train_loader))} batches (70%)")

    # Create model
    model = EidosProbableIMDB(
        vocab_size=len(train_dataset.vocab),
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=256,
        num_classes=2,
        dropout=0.0,
                use_overhead=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train with tension tracking
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch}/{args.epochs} - Tracking Structural Tension")
        print('=' * 70)

        log_file = output_dir / f'tension_epoch{epoch}.json'

        metrics = train_with_tension_tracking(
            model, train_loader, optimizer, criterion,
            device, epoch, log_file
        )

        # Analyze tension evolution
        plot_file = output_dir / f'tension_epoch{epoch}.png'
        analyze_tension_evolution(metrics['batch_metrics'], plot_file)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Train Acc:  {metrics['train_acc']:.2f}%")


if __name__ == "__main__":
    main()
