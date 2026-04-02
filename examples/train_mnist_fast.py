"""
EIDOS MNIST FAST BENCHMARK (Probable MEASUREMENT)

Implements a "ProbableConv2d" layer:
- Candidate Generation: 9 parallel convolutional kernels per layer.
- Observation: 1x1 Convolutional Observer measures local context.
- Collapse: Pixel-wise Hard Gumbel Softmax selects the best reality.

Architecture:
Input -> ProbableConv(32) -> Pool -> ProbableConv(64) -> Pool -> ProbableConv(64) -> Pool -> ProbableCollapse(Dense) -> Out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
import time

# Ensure eidos_nn export path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eidos_nn.layers.eidos_transform import eidosTransform
from eidos_nn.layers.convolution import ModularPhaseNorm2d
from eidos_nn.optim.fractal_optimizer import FractalOptimizer, FractalScheduler
from eidos_nn.models.eidos_measurement_driven import ProbableCollapseLayer

from eidos_nn.utils.measure_structural_tension import measure_path_tension
from eidos_nn.utils.certainty_validity import compute_cvs_from_logits


class ProbableConv2d(nn.Module):
    """
    Spatial Probable Collapse Layer.

    1. Generates 'num_paths' potential spatial features (Kernels).
    2. Measures context via 1x1 Convolutional Observer.
    3. Collapses to single feature map via Pixel-wise Selection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        num_paths=9,
        gumbel_tau=1.0,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.gumbel_tau = gumbel_tau

        # 1. Candidate Generation (Parallel Paths)
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                for _ in range(num_paths)
            ]
        )

        # 2. Observer (Measurement)
        self.observer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.Tanh(),
            nn.Conv2d(out_channels // 2, num_paths, 1),
        )

        # Eidos Principle: Use ModularPhaseNorm2d for structural normalization
        self.norm = ModularPhaseNorm2d(out_channels)

    def forward(self, x):
        # 1. Candidate Generation
        path_outputs = [k(x) for k in self.kernels]
        stacked_paths = torch.stack(path_outputs, dim=-1)

        # 2. Observation
        observer_scores = self.observer(x)

        # 3. Collapse
        if self.training:
            scores_perm = observer_scores.permute(0, 2, 3, 1)
            weights = F.gumbel_softmax(
                scores_perm, tau=self.gumbel_tau, hard=True, dim=-1
            )
        else:
            indices = torch.argmax(observer_scores, dim=1)
            weights = F.one_hot(indices, num_classes=self.num_paths).float()

        # Expand weights for broadcasting
        weights = weights.unsqueeze(1)

        # Collapse
        out = (stacked_paths * weights).sum(dim=-1)

        return self.norm(F.relu(out))


class EidosProbableMNIST(nn.Module):
    def __init__(self, num_paths=27):
        super().__init__()

        # Block 1
        self.conv1 = ProbableConv2d(1, 32, num_paths=num_paths)
        self.pool = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2 = ProbableConv2d(32, 64, num_paths=num_paths)

        # Block 3
        self.conv3 = ProbableConv2d(64, 32, num_paths=num_paths)

        # Classifier
        self.flatten_dim = 32 * 3 * 3

        # Use Eidos Probable Layer for final reasoning
        self.classifier = ProbableCollapseLayer(self.flatten_dim, num_paths=num_paths, dropout=0.0)
        self.final_proj = eidosTransform(self.flatten_dim, 10, num_rotation_planes=4)

    def forward(self, x):
        x = self.pool(self.conv1(x))  # -> [B, 24, 14, 14]
        x = self.pool(self.conv2(x))  # -> [B, 48, 7, 7]
        x = self.pool(self.conv3(x))  # -> [B, 24, 3, 3]

        x = x.view(-1, self.flatten_dim)

        # Probable Reasoning Step (Dense)
        x_seq = x.unsqueeze(1)
        x_seq = self.classifier(x_seq)
        x = x_seq.squeeze(1)

        x = self.final_proj(x)
        return x


def main():
    print(f"\n{'='*60}")
    print("EIDOS X MNIST: Probable SPEED RUN (ModularPhaseNorm + Metrics)")
    print(f"{'='*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Dataset
    print("Loading MNIST...", end="")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Num workers 0 for safety on Windows
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    print(" Done.")

    # Model
    model = EidosProbableMNIST(num_paths=9).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    import argparse
    parser = argparse.ArgumentParser(description="EIDOS MNIST fast benchmark")
    parser.add_argument('--csst-log', type=str, help='Path to CSST CSV log file')
    parser.add_argument('--train-log', type=str, help='Path to standard training CSV log')
    args = parser.parse_args()

    # CSST Logging Setup (Watcher - Fast Phase Plotting)
    csst_file = None
    if args.csst_log:
        os.makedirs(os.path.dirname(args.csst_log), exist_ok=True)
        csst_file = open(args.csst_log, 'w')
        # STRICT FORMAT for Watcher: step, avg_entropy, avg_tension, lock_rate_entropy
        csst_file.write("step,avg_entropy,avg_tension,lock_rate_entropy\n")

    # Training Log Setup (Detailed metrics)
    train_log_file = None
    if args.train_log:
        os.makedirs(os.path.dirname(args.train_log), exist_ok=True)
        train_log_file = open(args.train_log, 'w')
        train_log_file.write("epoch,batch,loss,train_acc,cvs,coverage\n")

    # CVS Log Setup
    cvs_log_path = "logs/mnist_fast/cvs_log.csv"
    os.makedirs(os.path.dirname(cvs_log_path), exist_ok=True)
    with open(cvs_log_path, 'w') as f:
        f.write("epoch,acc,commit_acc,approp_uncert,coverage,cvs\n")

    # Optimizer (Fractal)
    # Scale based on batch size (64) vs base (8) per Eidos axioms
    batch_scale = 64.0 / 8.0
    optimizer = FractalOptimizer(model.parameters(), base_lr=0.001, batch_scale=batch_scale)
    scheduler = FractalScheduler(optimizer, warmup_batches=50)
    criterion = nn.CrossEntropyLoss()

    # --- EIDOS LOGGER INTEGRATION ---
    from eidos_nn.utils.logger import eidosLogger
    config = {
        "model_name": "EidosProbableMNIST",
        "params": sum(p.numel() for p in model.parameters()),
        "dataset": "MNIST",
        "optimizer": "FractalOptimizer",
        "lr": 0.001,
        "num_epochs": 5,
        "batch_size": 64,
        "num_paths": 9
    }
    logger = eidosLogger("mnist_fast_benchmark", config, log_dir="logs/mnist_fast")
    # -------------------------------

    print("\nStarting Training (5 Epochs)...")

    global_step = 0
    best_acc = 0.0

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        start_time = time.time()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            tension = measure_path_tension(model, inputs, device)
            optimizer.adapt_frequencies(tension)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                current_acc = 100 * running_correct / running_total if running_total > 0 else 0
                logger.log_progress(epoch+1, i, len(trainloader), loss.item(), current_acc)

            # Manual CSST Logging (Watcher) - Kept for compatibility if args passed
            if csst_file:
                cv_result = compute_cvs_from_logits(outputs, labels, threshold=0.7)
                lock_rate = cv_result.Coverage
                csst_file.write(f"{global_step},{loss.item():.4f},{tension:.6f},{lock_rate:.6f}\n")
                if i % 10 == 0: csst_file.flush()

        # Training Stats
        avg_loss = running_loss / len(trainloader)
        train_acc = 100 * running_correct / running_total
        
        # Test Loop (with CVS aggregation)
        model.eval()
        correct = 0
        total = 0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Collect for CVS
                all_logits.append(outputs)
                all_labels.append(labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        # Compute CVS Metrics
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        cvs_result = compute_cvs_from_logits(all_logits, all_labels, threshold=0.7)
        
        # Log CVS
        with open(cvs_log_path, 'a') as f:
            f.write(f"{epoch+1},{test_acc:.4f},{cvs_result.CommitAcc:.4f},{cvs_result.AppropUncert:.4f},{cvs_result.Coverage:.4f},{cvs_result.CVS:.4f}\n")

        print(f"Epoch {epoch+1} | Time: {epoch_time:.1f}s | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | CVS: {cvs_result.CVS:.4f}")

        # Log Epoch
        logger.log_epoch({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "time": epoch_time
        })

        if test_acc > best_acc:
            best_acc = test_acc
            # Ensure directory exists
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/eidos_mnist_best.pth")

        # Save latest checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
        }, f"checkpoints/eidos_mnist_latest.pth")

        # Save training history JSON
        history_path = "checkpoints/training_history.json"
        with open(history_path, 'w') as f:
            # We can use the logger's history or build a simple one
            # The logger history is comprehensive, so let's dump that
            json.dump(logger.history, f, indent=2)

    # Finalize
    logger.finalize({
        "final_acc": best_acc,
        "completed": True
    })

    if csst_file: csst_file.close()
    if train_log_file: train_log_file.close()


if __name__ == "__main__":
    main()
