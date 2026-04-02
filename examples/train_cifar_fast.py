"""
EIDOS CIFAR-10 FAST BENCHMARK (Probable REASONING)

Implements a "ProbableConv2d" layer:
- Candidate Generation: 9 parallel convolutional kernels per layer.
- Observation: 1x1 Convolutional Observer measures local context.
- Collapse: Pixel-wise Hard Gumbel Softmax selects the best reality.

Architecture:
Input -> ProbableConv(32) -> Pool -> ProbableConv(64) -> Pool -> ProbableCollapse(Dense) -> Out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
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
from eidos_nn.layers.form_first_color import FormFirstColorLayer
from eidos_nn.utils.ablation_norms import IdentityNorm, RMSNorm2d
from eidos_nn.utils.logger import eidosLogger

from eidos_nn.utils.measure_structural_tension import measure_path_tension
from eidos_nn.utils.certainty_validity import compute_cvs_from_logits

class ProbableConv2d(nn.Module):
    """
    Spatial Probable Collapse Layer.
    
    1. Generates 'num_paths' potential spatial features (Kernels).
    2. Measures context via 1x1 Convolutional Observer.
    3. Collapses to single feature map via Pixel-wise Selection.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_paths=9, gumbel_tau=1.0, norm_type="modular"):
        super().__init__()
        self.num_paths = num_paths
        self.gumbel_tau = gumbel_tau
        
        # 1. Candidate Generation (Parallel Paths)
        # 9 Independent Kernels representing different spatial interpretations
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            for _ in range(num_paths)
        ])
        
        # 2. Observer (Measurement)
        # Lightweight 1x1 conv to score paths based on input context
        self.observer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1), # Bottleneck
            nn.Tanh(), # Stability
            nn.Conv2d(out_channels // 2, num_paths, 1) # Score per path
        )
        
        if norm_type == "modular":
            self.norm = ModularPhaseNorm2d(out_channels)
        elif norm_type == "rms":
            self.norm = RMSNorm2d(out_channels)
        elif norm_type == "none":
            self.norm = IdentityNorm()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        
    def forward(self, x):
        # x: [batch, in_c, h, w]
        
        # 1. Candidate Generation
        # [batch, out_c, h, w, num_paths]
        # Stack results of all kernels
        path_outputs = [k(x) for k in self.kernels]
        stacked_paths = torch.stack(path_outputs, dim=-1)
        
        # 2. Observation
        # [batch, num_paths, h, w]
        observer_scores = self.observer(x)
        
        # 3. Collapse
        if self.training:
            # Gumbel-Softmax (Hard) over num_paths dim
            # scores: [B, P, H, W] -> permute to [B, H, W, P] for gumbel
            scores_perm = observer_scores.permute(0, 2, 3, 1)
            weights = F.gumbel_softmax(scores_perm, tau=self.gumbel_tau, hard=True, dim=-1) # [B, H, W, P]
        else:
            # Hard Argmax
            indices = torch.argmax(observer_scores, dim=1) # [B, H, W]
            weights = F.one_hot(indices, num_classes=self.num_paths).float() # [B, H, W, P]
            
        # Expand weights for broadcasting: [B, H, W, P] -> [B, 1, H, W, P] (broadcasting over out_channels)
        weights = weights.unsqueeze(1)
        
        # Collapse
        # stacked: [B, C, H, W, P]
        # weights: [B, 1, H, W, P]
        out = (stacked_paths * weights).sum(dim=-1)
        
        return self.norm(F.relu(out)) # Using ReLU for speed, could switch to Tanh if requested

class EidosProbableCNN(nn.Module):
    def __init__(self, use_color_frontend=False, matcher_mode="geometric", matcher_collapse="truncate", norm_type="modular"):
        super().__init__()
        self.use_color_frontend = use_color_frontend
        self.matcher_mode = matcher_mode
        self.matcher_collapse = matcher_collapse

        # Optional form-first color frontend. Off by default for CIFAR-10.
        self.frontend = None
        if self.use_color_frontend:
            self.frontend = FormFirstColorLayer(
                use_cmyk=False,
                color_depth=4,
                adaptive_depth=True,
                use_gridnorm=True,
                assume_normalized=False
            )
        
        # Block 1 - Expecting 3 Channels (RGB from frontend)
        self.conv1 = ProbableConv2d(3, 32, num_paths=9, norm_type=norm_type)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = ProbableConv2d(32, 64, num_paths=9, norm_type=norm_type)
        
        # Block 3
        self.conv3 = ProbableConv2d(64, 64, num_paths=9, norm_type=norm_type)
        
        # Classifier
        # Flatten: 64 * 4 * 4 (assuming 32x32 input -> 16 -> 8 -> 4)
        self.flatten_dim = 64 * 4 * 4
        
        # Use Eidos Probable Layer for final reasoning
        self.classifier = ProbableCollapseLayer(
            self.flatten_dim,
            num_paths=9,
            dropout=0.0,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )
        self.final_proj = eidosTransform(
            self.flatten_dim,
            10,
            num_rotation_planes=4,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )

    def forward(self, x):
        # Input: [B, 3, 32, 32] [0, 1]
        if self.frontend is not None:
            x = self.frontend(x)
        
        x = self.pool(self.conv1(x)) # -> [B, 32, 16, 16]
        x = self.pool(self.conv2(x)) # -> [B, 64, 8, 8]
        x = self.pool(self.conv3(x)) # -> [B, 64, 4, 4]
        
        x = x.view(-1, self.flatten_dim)
        
        # Probable Reasoning Step (Dense)
        # Adds "seq" dimension for compatibility: [B, 1, Dim]
        x_seq = x.unsqueeze(1)
        x_seq = self.classifier(x_seq)
        x = x_seq.squeeze(1)
        
        x = self.final_proj(x)
        return x

def main():
    print(f"\n{'='*60}")
    print("EIDOS X CIFAR-10: Probable SPEED RUN (ModularPhaseNorm + Metrics)")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) <-- REMOVED for FormFirst
    ])
    
    import argparse
    parser = argparse.ArgumentParser(description="EIDOS CIFAR-10 fast benchmark")
    parser.add_argument('--csst-log', type=str, help='Path to CSST CSV log file')
    parser.add_argument('--train-log', type=str, help='Path to standard training CSV log')
    parser.add_argument('--color-frontend', choices=['on', 'off'], default='off',
                        help='Enable FormFirstColorLayer frontend')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--matcher-mode', choices=['geometric', 'canonical'], default='geometric',
                        help='Dimension matcher backend for eidosTransform bridges')
    parser.add_argument('--matcher-collapse', choices=['truncate', 'fold'], default='truncate',
                        help='Geometric compression collapse rule (ignored for canonical matcher)')
    parser.add_argument('--norm-type', choices=['modular', 'rms', 'none'], default='modular',
                        help='Normalization ablation for conv blocks')
    args = parser.parse_args()
    use_color_frontend = args.color_frontend == 'on'

    # Dataset
    print("Loading CIFAR-10...", end="")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Num workers 0 for safety on Windows
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    print(" Done.")
    
    # Model
    model = EidosProbableCNN(
        use_color_frontend=use_color_frontend,
        matcher_mode=args.matcher_mode,
        matcher_collapse=args.matcher_collapse,
        norm_type=args.norm_type,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Color frontend: {'ON' if use_color_frontend else 'OFF'}")
    print(f"Norm type: {args.norm_type.upper()}")
    print(f"Model Parameters: {total_params/1e6:.2f}M")

    logger = eidosLogger(
        run_name="cifar_fast_benchmark",
        config={
            "model_name": "EidosProbableCNN",
            "params": total_params,
            "dataset": "CIFAR-10",
            "optimizer": "FractalOptimizer",
            "lr": 0.002,
            "num_epochs": args.epochs,
            "batch_size": 128,
            "num_paths": 9,
            "color_frontend": use_color_frontend,
            "matcher_mode": args.matcher_mode,
            "matcher_collapse": args.matcher_collapse,
            "norm_type": args.norm_type,
        },
        log_dir="logs/cifar_fast",
    )

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

    # Optimizer (Fractal)
    # Scale based on batch size (128) vs base (8) to ensure differentiation
    # 128 / 8 = 16.0 -> Coarse (x4), Triadic (~x2.3), Fine (~x1.7)
    batch_scale = 128.0 / 8.0 
    optimizer = FractalOptimizer(model.parameters(), base_lr=0.002, batch_scale=batch_scale)
    scheduler = FractalScheduler(optimizer, warmup_batches=50) # Shorter warmup
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting Training ({args.epochs} Epochs)...")
    
    global_step = 0
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        chunk_loss = 0.0
        chunk_correct = 0
        chunk_total = 0
        start_time = time.time()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Vital: Adapt Fractal Optimizer based on Structural Tension
            # Using official measure_path_tension (output variance/path coordination)
            tension = measure_path_tension(model, inputs, device)
            optimizer.adapt_frequencies(tension)

            optimizer.step()
            scheduler.step(structural_tension=tension)

            epoch_loss += loss.item()
            chunk_loss += loss.item()
            global_step += 1

            # Measure Train Acc & CV Metrics
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            epoch_total += labels.size(0)
            epoch_correct += batch_correct
            chunk_total += labels.size(0)
            chunk_correct += batch_correct

            # Calc metrics
            current_acc = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0

            # Use Rigorous Certainty-Validity Metrics
            cv_result = compute_cvs_from_logits(outputs, labels, threshold=0.7)
            lock_rate = cv_result.Coverage

            # 1. Fast Log (For Watcher/Phase Plot)
            if csst_file:
                # Map metrics to what the plotter expects:
                # epoch -> global_step (continuous time)
                # avg_entropy -> loss (system entropy)
                # avg_tension -> tension (REAL structural tension from gradients)
                # lock_rate_entropy -> lock_rate (CVS Coverage)
                csst_file.write(f"{global_step},{loss.item():.4f},{tension:.6f},{lock_rate:.6f}\n")
                if i % 10 == 0:
                    csst_file.flush()

            # 2. Detailed Training Log
            if train_log_file:
                train_log_file.write(f"{epoch+1},{i+1},{loss.item():.4f},{current_acc:.2f},{cv_result.CVS:.4f},{lock_rate:.4f}\n")
                if i % 10 == 0:
                    train_log_file.flush()

            if i % 100 == 99:
                avg_loss = chunk_loss / 100
                avg_acc = 100 * chunk_correct / chunk_total
                print(f"[{epoch+1}, {i+1:3d}] Loss: {avg_loss:.3f} | Train Acc: {avg_acc:.2f}%")
                logger.log_progress(epoch + 1, i + 1, len(trainloader), avg_loss, avg_acc)
                chunk_loss = 0.0
                chunk_correct = 0
                chunk_total = 0

        # Test loop runs every epoch, independent of optional log files.
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += loss.item()

                # Collect for CVS
                all_logits.append(outputs.cpu())
                all_labels.append(labels.cpu())

        acc = 100 * correct / total
        train_acc = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        train_loss = epoch_loss / len(trainloader)
        avg_test_loss = test_loss / len(testloader)
        epoch_time = time.time() - start_time

        # Calculate Gap using full-epoch train accuracy.
        gap = acc - train_acc

        # Compute CVS
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        cv_result = compute_cvs_from_logits(all_logits, all_labels, threshold=0.7)

        print(f"Epoch {epoch+1} Completed in {epoch_time:.1f}s | Test Acc: {acc:.2f}% (Gap: {gap:+.2f}%) | CVS: {cv_result.CVS:.4f}")
        print(f"         CVS Matrix: CC={cv_result.CC:5d} CI={cv_result.CI:5d} UC={cv_result.UC:5d} UI={cv_result.UI:5d}")
        print(f"         CommitAcc: {cv_result.CommitAcc*100:.2f}% | AppropUncert: {cv_result.AppropUncert*100:.2f}% | Coverage: {cv_result.Coverage*100:.2f}%")

        logger.log_epoch({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(avg_test_loss, 4),
            "test_acc": round(acc, 2),
            "gap": round(gap, 2),
        })
        logger.log_timing(epoch + 1, {"epoch_seconds": round(epoch_time, 2)})
        best_acc = max(best_acc, acc)

    if csst_file:
        csst_file.close()
    if train_log_file:
        train_log_file.close()
    logger.finalize({"best_test_acc": round(best_acc, 2)})
    print(f"Logs saved to: {logger.json_path}")

if __name__ == "__main__":
    main()
