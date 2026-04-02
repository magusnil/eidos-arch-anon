"""
Dataset 1 V7 MLP baselines under the fixed order-preserving encoding.

Supports:
1. full-input MLP baseline (Table 1 replacement)
2. per-choice MLP choice scorer (Table 2 replacement)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class OrderedByteEncoder:
    def __init__(self, max_bytes: int = 512):
        self.max_bytes = max_bytes
        self.input_dim = max_bytes * 2

    def encode(self, text: str) -> np.ndarray:
        encoded = text.encode("utf-8", errors="ignore")[: self.max_bytes]
        features = np.zeros((self.max_bytes, 2), dtype=np.float32)
        for idx, byte_value in enumerate(encoded):
            features[idx, 0] = (byte_value + 1) / 256.0
            features[idx, 1] = 1.0
        return features.reshape(-1)


def split_choice_sample(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    context_lines: List[str] = []
    choices: List[str] = []
    query_line = None
    in_choices = False

    for line in lines:
        if line.startswith("[CHOICES]"):
            in_choices = True
            continue
        if line.startswith("[QUERY]"):
            in_choices = False
            query_line = line
            continue
        if in_choices:
            choices.append(line)
        else:
            context_lines.append(line)

    if query_line is None or len(choices) != 3:
        return None

    return "\n".join(context_lines), choices, query_line


class Dataset1V7MLPDataset(Dataset):
    def __init__(self, path: str, max_bytes: int = 512):
        self.encoder = OrderedByteEncoder(max_bytes=max_bytes)
        self.features: List[np.ndarray] = []
        self.labels: List[int] = []

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.features.append(self.encoder.encode(obj["input"]))
                self.labels.append(int(obj["label"]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class Dataset1V7ChoiceScorerDataset(Dataset):
    def __init__(self, path: str, max_bytes: int = 512):
        self.encoder = OrderedByteEncoder(max_bytes=max_bytes)
        self.features: List[np.ndarray] = []
        self.labels: List[int] = []

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                parsed = split_choice_sample(obj["input"])
                if parsed is None:
                    continue
                context, choices, query = parsed
                label = int(obj["label"])
                if label < 0 or label >= len(choices):
                    continue
                choice_features = []
                for choice in choices:
                    blob = f"{context}\n[CHOICE] {choice}\n{query}\n"
                    choice_features.append(self.encoder.encode(blob))
                self.features.append(np.stack(choice_features, axis=0))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPChoiceScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, choices, dims = x.shape
        flat = x.view(batch * choices, dims)
        return self.scorer(flat).view(batch, choices)


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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--choice-scorer", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.choice_scorer:
        train_dataset = Dataset1V7ChoiceScorerDataset(args.train, max_bytes=args.max_bytes)
        val_dataset = Dataset1V7ChoiceScorerDataset(args.val, max_bytes=args.max_bytes)
        input_dim = train_dataset.encoder.input_dim
        model = MLPChoiceScorer(input_dim=input_dim, hidden_dim=args.hidden_dim)
        title = "=== DATASET 1 V7 MLP CHOICE SCORER (ORDERED BYTES) ==="
    else:
        train_dataset = Dataset1V7MLPDataset(args.train, max_bytes=args.max_bytes)
        val_dataset = Dataset1V7MLPDataset(args.val, max_bytes=args.max_bytes)
        input_dim = train_dataset.encoder.input_dim
        model = MLPBaseline(input_dim=input_dim, hidden_dim=args.hidden_dim)
        title = "=== DATASET 1 V7 MLP BASELINE (ORDERED BYTES) ==="

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(title)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim} | Hidden dim: {args.hidden_dim}")

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
