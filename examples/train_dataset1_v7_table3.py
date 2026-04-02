"""
Dataset 1 V7 Table 3 redo under the fixed order-preserving encoding.

This is the decomplexed three-branch R1 choice-scorer ablation:
- learned selection over {-W, 0, +W}
- no PathBundle preservation across depth
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys_path_root = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(sys_path_root))

from train_dataset1_v7_mlp_baselines import Dataset1V7ChoiceScorerDataset


class R1SelectionLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, scorer_hidden: int = 64, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.w_base = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.b_base = nn.Parameter(torch.zeros(out_features))
        self.scorer = nn.Sequential(
            nn.Linear(out_features, scorer_hidden),
            nn.Tanh(),
            nn.Linear(scorer_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_neg = x @ (-self.w_base) + (-self.b_base)
        out_zero = torch.zeros_like(out_neg)
        out_pos = x @ self.w_base + self.b_base
        stacked = torch.stack([out_neg, out_zero, out_pos], dim=1)
        scores = self.scorer(stacked).squeeze(-1)

        if self.training:
            weights = F.gumbel_softmax(scores, tau=self.tau, hard=True, dim=1)
        else:
            indices = torch.argmax(scores, dim=1)
            weights = F.one_hot(indices, num_classes=3).float()

        return torch.sum(stacked * weights.unsqueeze(-1), dim=1)


class Dataset1V7Table3Redo(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, scorer_hidden: int = 64, tau: float = 1.0):
        super().__init__()
        self.r1_1 = R1SelectionLinear(input_dim, hidden_dim, scorer_hidden=scorer_hidden, tau=tau)
        self.r1_2 = R1SelectionLinear(hidden_dim, hidden_dim, scorer_hidden=scorer_hidden, tau=tau)
        self.head = R1SelectionLinear(hidden_dim, 1, scorer_hidden=max(8, scorer_hidden // 2), tau=tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, choices, dims = x.shape
        flat = x.view(batch * choices, dims)
        h = torch.tanh(self.r1_1(flat))
        h = torch.tanh(self.r1_2(h))
        return self.head(h).view(batch, choices)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--scorer-hidden", type=int, default=64)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)

    train_dataset = Dataset1V7ChoiceScorerDataset(args.train, max_bytes=args.max_bytes)
    val_dataset = Dataset1V7ChoiceScorerDataset(args.val, max_bytes=args.max_bytes)
    input_dim = train_dataset.encoder.input_dim

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = Dataset1V7Table3Redo(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        scorer_hidden=args.scorer_hidden,
        tau=args.tau,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("=== DATASET 1 V7 TABLE 3 REDO (ORDERED BYTES) ===")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim} | Hidden dim: {args.hidden_dim} | Tau: {args.tau}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        gap = val_acc - train_acc
        print(
            f"Epoch {epoch:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"Gap: {gap:+.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
