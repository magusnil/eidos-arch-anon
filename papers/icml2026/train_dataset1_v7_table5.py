"""
Dataset 1 V7 Table 5 redo using the current public-release architecture.

This runner is intentionally separate from older experimental harnesses.
It follows the public-release template style:
1. deterministic setup
2. current Eidos model construction
3. Fractal optimizer + scheduler
4. structured JSON logging
5. optional per-run system profiling
"""

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PUBLIC_RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "core"))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "layers"))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "optim"))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "utils"))

from path_bundle import PathBundle
from eidos_transform import eidosTransform, eidosSequential
from fractal_optimizer import FractalOptimizer, FractalScheduler
from logger import eidosLogger
from modular_phase_norm import ModularPhaseNorm


class Dataset1V7ChoiceDataset(Dataset):
    def __init__(self, path: str, max_bytes: int = 512):
        self.features: List[np.ndarray] = []
        self.labels: List[int] = []
        self.max_bytes = max_bytes
        self.input_dim = max_bytes * 2

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                parsed = self._split_sample(obj.get("input", ""))
                if parsed is None:
                    continue
                context, choices, query = parsed
                label = int(obj.get("label", -1))
                if label < 0 or label >= len(choices):
                    continue

                choice_features = []
                for choice in choices:
                    blob = f"{context}\n[CHOICE] {choice}\n{query}\n"
                    choice_features.append(self._encode_blob(blob))

                if len(choice_features) != 3:
                    continue

                self.features.append(np.stack(choice_features, axis=0))
                self.labels.append(label)

        if not self.features:
            raise RuntimeError(f"No usable samples parsed from {path}")

    def _encode_blob(self, blob: str) -> np.ndarray:
        """
        Order-preserving byte encoding.

        Each byte position contributes two features:
        1. normalized byte value in [0, 1] with +1 offset so pad can stay zero
        2. validity mask so padding is explicitly represented
        """
        encoded = blob.encode("utf-8", errors="ignore")[: self.max_bytes]
        features = np.zeros((self.max_bytes, 2), dtype=np.float32)
        for idx, byte_value in enumerate(encoded):
            features[idx, 0] = (byte_value + 1) / 256.0
            features[idx, 1] = 1.0
        return features.reshape(-1)

    @staticmethod
    def _split_sample(text: str):
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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class SetValuedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, bundle: PathBundle) -> PathBundle:
        batch_size = bundle.batch_size
        output_paths = []
        quality_score_list = []

        for path_idx in range(bundle.num_paths):
            input_path = bundle.get_path(path_idx)
            negative_path = F.linear(input_path, -self.weight, self.bias)
            positive_path = F.linear(input_path, self.weight, self.bias)
            if self.bias is not None:
                zero_path = self.bias.unsqueeze(0).expand(batch_size, -1)
            else:
                zero_path = torch.zeros(
                    batch_size,
                    self.out_features,
                    device=bundle.device,
                    dtype=bundle.dtype,
                )

            output_paths.extend([negative_path, zero_path, positive_path])
            input_quality = bundle.quality_scores[:, path_idx:path_idx + 1] / 3.0
            quality_score_list.extend([input_quality, input_quality, input_quality])

        output_data = torch.stack(output_paths, dim=2)
        output_scores = torch.cat(quality_score_list, dim=1)
        return PathBundle(output_data, quality_scores=output_scores)


class PathBundleTanh(nn.Module):
    def forward(self, bundle: PathBundle) -> PathBundle:
        outputs = []
        for path_idx in range(bundle.num_paths):
            outputs.append(torch.tanh(bundle.get_path(path_idx)).unsqueeze(-1))
        return PathBundle(torch.cat(outputs, dim=-1), quality_scores=bundle.quality_scores)


class PathQualityScorer(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int = 64,
        scorer_mode: str = "modular_phase",
        matcher_mode: str = "geometric",
        matcher_collapse: str = "fold",
    ):
        super().__init__()
        first = eidosTransform(
            features,
            hidden,
            num_rotation_planes=2,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )
        second = eidosTransform(
            hidden,
            1,
            num_rotation_planes=1,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )

        if scorer_mode == "legacy_tanh":
            middle = nn.Tanh()
        elif scorer_mode == "modular_phase":
            middle = ModularPhaseNorm(hidden, base=7)
        else:
            raise ValueError(f"Unknown scorer_mode: {scorer_mode}")

        self.net = eidosSequential(first, middle, second)

    def forward(self, bundle: PathBundle) -> torch.Tensor:
        x = bundle.data.permute(0, 2, 1)
        return self.net(x).squeeze(-1)


class Dataset1V7Table5Redo(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_paths: int = 9,
        scorer_hidden: int = 64,
        scorer_mode: str = "modular_phase",
        matcher_mode: str = "geometric",
        matcher_collapse: str = "fold",
    ):
        super().__init__()
        self.max_paths = max_paths
        self.layers = nn.ModuleList()
        self.scorers = nn.ModuleList()
        self.activation = PathBundleTanh()

        dims = [input_dim] + [hidden_dim] * num_layers
        for idx in range(num_layers):
            self.layers.append(SetValuedLinear(dims[idx], dims[idx + 1]))
            self.scorers.append(
                PathQualityScorer(
                    dims[idx + 1],
                    hidden=scorer_hidden,
                    scorer_mode=scorer_mode,
                    matcher_mode=matcher_mode,
                    matcher_collapse=matcher_collapse,
                )
            )

        self.output_layer = SetValuedLinear(hidden_dim, 1)
        self.output_scorer = PathQualityScorer(
            1,
            hidden=max(8, scorer_hidden // 2),
            scorer_mode=scorer_mode,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )

    def _score_and_prune(self, bundle: PathBundle, scorer: nn.Module) -> PathBundle:
        scores = scorer(bundle)
        weights = F.softmax(scores, dim=1)
        bundle.set_quality_scores(weights)
        if bundle.num_paths > self.max_paths:
            bundle = bundle.prune(self.max_paths)
            denom = bundle.quality_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
            bundle.set_quality_scores(bundle.quality_scores / denom)
        return bundle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, choices, dims = x.shape
        flat = x.view(batch * choices, dims)
        bundle = PathBundle(flat, num_paths=1)

        for layer, scorer in zip(self.layers, self.scorers):
            bundle = layer(bundle)
            bundle = self.activation(bundle)
            bundle = self._score_and_prune(bundle, scorer)

        bundle = self.output_layer(bundle)
        bundle = self._score_and_prune(bundle, self.output_scorer)
        return bundle.collapse("weighted").view(batch, choices)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def maybe_start_profiler(args) -> subprocess.Popen | None:
    if not args.profile_system:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = args.profile_output or str(
        Path(__file__).resolve().parent / "logs" / f"train_dataset1_v7_table5_{timestamp}.system.jsonl"
    )
    cmd = [
        sys.executable,
        str(PUBLIC_RELEASE_ROOT / "profile_system_usage.py"),
        "--interval",
        str(args.profile_interval),
        "--output",
        output,
        "--stop-on-pid",
        str(Path().resolve().stat().st_ino if False else 0),
    ]
    cmd[-1] = str(os_getpid())
    if args.profile_baseline_samples > 0:
        cmd.extend(["--baseline-samples", str(args.profile_baseline_samples)])
    print(f"System profiler: {output}")
    return subprocess.Popen(cmd)


def os_getpid() -> int:
    import os

    return os.getpid()


def run_epoch(model, loader, optimizer, scheduler, criterion, device, train: bool, logger=None, epoch=0):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    iterator = loader
    if train:
        from tqdm import tqdm

        iterator = tqdm(loader, desc=f"Training Epoch {epoch}")
    else:
        from tqdm import tqdm

        iterator = tqdm(loader, desc="Evaluating")

    with torch.set_grad_enabled(train):
        for batch_idx, (batch_x, batch_y) in enumerate(iterator):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if train:
                loss.backward()
                optimizer.adapt_frequencies(0.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            acc = 100.0 * correct / max(total, 1)

            iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

            if logger is not None and train and batch_idx % 50 == 0:
                logger.log_progress(epoch, batch_idx, len(loader), loss.item(), acc)

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=9)
    parser.add_argument("--scorer-hidden", type=int, default=64)
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--scorer-mode", choices=["legacy_tanh", "modular_phase"], default="modular_phase")
    parser.add_argument("--matcher-mode", choices=["geometric", "canonical"], default="geometric")
    parser.add_argument("--matcher-collapse", choices=["truncate", "fold"], default="fold")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--profile-system", action="store_true")
    parser.add_argument("--profile-interval", type=float, default=2.0)
    parser.add_argument("--profile-baseline-samples", type=int, default=0)
    parser.add_argument("--profile-output", type=str, default=None)
    args = parser.parse_args()

    set_reproducibility(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_dataset = Dataset1V7ChoiceDataset(args.train, max_bytes=args.max_bytes)
    val_dataset = Dataset1V7ChoiceDataset(args.val, max_bytes=args.max_bytes)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = Dataset1V7Table5Redo(
        input_dim=train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_paths=args.max_paths,
        scorer_hidden=args.scorer_hidden,
        scorer_mode=args.scorer_mode,
        matcher_mode=args.matcher_mode,
        matcher_collapse=args.matcher_collapse,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger = eidosLogger(
        run_name=f"dataset1_v7_table5_h{args.hidden_dim}",
        config={
            "model_name": "Dataset1V7Table5Redo",
            "dataset": "dataset_1_v7",
            "encoding": "ordered_byte_sequence_with_mask",
            "max_bytes": args.max_bytes,
            "input_dim": train_dataset.input_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "max_paths": args.max_paths,
            "scorer_hidden": args.scorer_hidden,
            "scorer_mode": args.scorer_mode,
            "matcher_mode": args.matcher_mode,
            "matcher_collapse": args.matcher_collapse,
            "optimizer": "FractalOptimizer",
            "seed": args.seed,
            "params": total_params,
        },
        log_dir=str(Path(__file__).resolve().parent / "logs"),
    )

    batch_scale = args.batch_size / 32.0
    optimizer = FractalOptimizer(model.parameters(), base_lr=args.lr, batch_scale=batch_scale)
    scheduler = FractalScheduler(optimizer, warmup_batches=5)
    criterion = nn.CrossEntropyLoss()

    profiler_proc = maybe_start_profiler(args)

    print("=== DATASET 1 V7 TABLE 5 REDO (PUBLIC RELEASE) ===")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Device: {device}")
    print(f"Batches/epoch: {len(train_loader)} | Batch size: {args.batch_size}")
    print(f"Model: Dataset1V7Table5Redo | Params: {total_params:,}")
    print(
        f"Encoding: ordered-bytes({args.max_bytes}) | "
        f"Scorer: {args.scorer_mode} | Matcher: {args.matcher_mode}/{args.matcher_collapse} | "
        f"Optimizer: fractal"
    )

    best_val = 0.0
    try:
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
                train=True,
                logger=logger,
                epoch=epoch,
            )
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                None,
                None,
                criterion,
                device,
                train=False,
            )
            elapsed = time.time() - start
            gap = val_acc - train_acc

            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"Gap:   {gap:+.2f}%")

            logger.log_epoch(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 2),
                    "gap_val_minus_train": round(gap, 2),
                }
            )
            logger.log_timing(epoch, {"epoch_seconds": round(elapsed, 2)})

            if val_acc > best_val:
                best_val = val_acc

        logger.finalize({"best_val_acc": round(best_val, 2)})
        print(f"\nBest Val Accuracy: {best_val:.2f}%")
        print(f"Logs saved to: {logger.json_path}")
    finally:
        if profiler_proc is not None:
            time.sleep(0.5)
            if profiler_proc.poll() is None:
                profiler_proc.terminate()


if __name__ == "__main__":
    main()
