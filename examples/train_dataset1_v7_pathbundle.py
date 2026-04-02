"""
Reproducible Dataset 1 V7 PathBundle choice scorer.

This runner preserves the older PathBundle choice-scoring line used for
Dataset 1 while avoiding both:
1. the old experiment scripts, and
2. the later measurement/FormSpace autoregression stack.

It exposes the scorer stack explicitly so reruns can distinguish between:
- legacy_tanh: eidosTransform -> nn.Tanh -> eidosTransform
- modular_phase: eidosTransform -> ModularPhaseNorm -> eidosTransform
"""

import argparse
import json
import os
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

PUBLIC_RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "core"))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "layers"))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "optim"))
sys.path.insert(0, str(PUBLIC_RELEASE_ROOT / "eidos_nn" / "utils"))
sys.stdout.reconfigure(line_buffering=True)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from path_bundle import PathBundle
from eidos_transform import eidosTransform, eidosSequential
from fractal_optimizer import FractalOptimizer, FractalScheduler
from modular_phase_norm import ModularPhaseNorm
from ablation_norms import IdentityNorm, RMSNorm1d
from eidos_nn.models.eidos_measurement_driven import ProbableCollapseLayer


class JsonlChoiceScorerDataset(Dataset):
    def __init__(self, path: str):
        self.features: List[np.ndarray] = []
        self.labels: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("input", "")
                label = int(obj.get("label", -1))
                parsed = self._split_sample(text)
                if parsed is None:
                    continue
                context, choices, query = parsed
                if label < 0 or label >= len(choices):
                    continue
                choice_feats = []
                for choice in choices:
                    blob = f"{context}\n[CHOICE] {choice}\n{query}\n"
                    counts = np.zeros(256, dtype=np.float32)
                    for b in blob.encode("utf-8", errors="ignore"):
                        counts[b] += 1.0
                    total = counts.sum()
                    if total > 0:
                        counts /= total
                    choice_feats.append(counts)
                if len(choice_feats) != 3:
                    continue
                self.features.append(np.stack(choice_feats, axis=0))
                self.labels.append(label)

        if not self.features:
            raise RuntimeError(f"No samples parsed from {path}")

    def _split_sample(self, text: str):
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
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
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

    def forward(self, x: PathBundle) -> PathBundle:
        batch_size = x.batch_size
        output_paths = []
        quality_score_list = []

        for p_idx in range(x.num_paths):
            input_path = x.get_path(p_idx)

            negative_path = F.linear(input_path, -self.weight, self.bias)
            positive_path = F.linear(input_path, self.weight, self.bias)
            if self.bias is not None:
                zero_path = self.bias.unsqueeze(0).expand(batch_size, -1)
            else:
                zero_path = torch.zeros(
                    batch_size,
                    self.out_features,
                    device=x.device,
                    dtype=x.dtype,
                )

            output_paths.extend([negative_path, zero_path, positive_path])

            input_quality = x.quality_scores[:, p_idx:p_idx + 1] / 3.0
            quality_score_list.extend([input_quality, input_quality, input_quality])

        output_data = torch.stack(output_paths, dim=2)
        output_scores = torch.cat(quality_score_list, dim=1)
        return PathBundle(output_data, quality_scores=output_scores)


class PathBundleTanh(nn.Module):
    def forward(self, x: PathBundle) -> PathBundle:
        new_paths = []
        for p in range(x.num_paths):
            new_paths.append(torch.tanh(x.get_path(p)).unsqueeze(-1))
        return PathBundle(torch.cat(new_paths, dim=-1), quality_scores=x.quality_scores)


class PathQualityScorer(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int = 64,
        scorer_mode: str = "modular_phase",
        matcher_mode: str = "canonical",
    ):
        super().__init__()
        first = eidosTransform(
            features,
            hidden,
            num_rotation_planes=2,
            matcher_mode=matcher_mode,
        )
        second = eidosTransform(
            hidden,
            1,
            num_rotation_planes=1,
            matcher_mode=matcher_mode,
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


class PathBundleChoiceScorer(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_paths: int = 9,
        scorer_hidden: int = 64,
        scorer_mode: str = "modular_phase",
        matcher_mode: str = "canonical",
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
                )
            )

        self.output_layer = SetValuedLinear(hidden_dim, 1)
        self.output_scorer = PathQualityScorer(
            1,
            hidden=max(8, scorer_hidden // 2),
            scorer_mode=scorer_mode,
            matcher_mode=matcher_mode,
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


class ProbableDataset1ChoiceScorer(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 128,
        num_layers: int = 2,
        num_paths: int = 9,
        matcher_mode: str = "geometric",
        matcher_collapse: str = "truncate",
        norm_type: str = "modular",
    ):
        super().__init__()
        self.input_proj = eidosTransform(
            input_dim,
            d_model,
            num_rotation_planes=2,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )
        self.layers = nn.ModuleList([
            ProbableCollapseLayer(
                d_model,
                num_paths=num_paths,
                dropout=0.0,
                gumbel_tau=1.0,
                matcher_mode=matcher_mode,
                matcher_collapse=matcher_collapse,
            )
            for _ in range(num_layers)
        ])
        if norm_type == "modular":
            self.norm = ModularPhaseNorm(d_model, base=7)
        elif norm_type == "rms":
            self.norm = RMSNorm1d(d_model)
        elif norm_type == "none":
            self.norm = IdentityNorm()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.final_proj = eidosTransform(
            d_model,
            1,
            num_rotation_planes=2,
            matcher_mode=matcher_mode,
            matcher_collapse=matcher_collapse,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, choices, features]
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.final_proj(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, device="cpu", use_tqdm=True):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    iterator = loader
    if use_tqdm and tqdm is not None:
        iterator = tqdm(loader, desc="Train" if is_train else "Val", leave=False)

    with torch.set_grad_enabled(is_train):
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                if isinstance(optimizer, FractalOptimizer):
                    optimizer.adapt_frequencies(0.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            if use_tqdm and tqdm is not None:
                acc = 100.0 * correct / max(total, 1)
                iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=9)
    parser.add_argument("--scorer-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--model-family", choices=["pathbundle", "probable"], default="pathbundle")
    parser.add_argument("--scorer-mode", choices=["legacy_tanh", "modular_phase"], default="modular_phase")
    parser.add_argument("--matcher-mode", choices=["canonical", "geometric"], default="canonical")
    parser.add_argument("--matcher-collapse", choices=["truncate", "fold"], default="truncate")
    parser.add_argument("--norm-type", choices=["modular", "rms", "none"], default="modular")
    parser.add_argument("--optimizer", choices=["adam", "fractal"], default="fractal")
    parser.add_argument("--profile-system", action="store_true")
    parser.add_argument("--profile-interval", type=float, default=2.0)
    parser.add_argument("--profile-baseline-samples", type=int, default=0)
    parser.add_argument("--profile-output", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    train_ds = JsonlChoiceScorerDataset(args.train)
    val_ds = JsonlChoiceScorerDataset(args.val)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.model_family == "pathbundle":
        model = PathBundleChoiceScorer(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            max_paths=args.max_paths,
            scorer_hidden=args.scorer_hidden,
            scorer_mode=args.scorer_mode,
            matcher_mode=args.matcher_mode,
        ).to(device)
    else:
        model = ProbableDataset1ChoiceScorer(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_paths=args.max_paths,
            matcher_mode=args.matcher_mode,
            matcher_collapse=args.matcher_collapse,
            norm_type=args.norm_type,
        ).to(device)

    scheduler = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        batch_scale = args.batch_size / 32.0
        optimizer = FractalOptimizer(model.parameters(), base_lr=args.lr, batch_scale=batch_scale)
        scheduler = FractalScheduler(optimizer, warmup_batches=5)
    criterion = nn.CrossEntropyLoss()

    profiler_proc = None
    if args.profile_system:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        profile_output = args.profile_output or str(
            Path(__file__).resolve().parent / "logs" / f"train_dataset1_v7_pathbundle_{timestamp}.system.jsonl"
        )
        profiler_cmd = [
            sys.executable,
            str(PUBLIC_RELEASE_ROOT / "profile_system_usage.py"),
            "--interval",
            str(args.profile_interval),
            "--output",
            profile_output,
            "--stop-on-pid",
            str(os.getpid()),
        ]
        if args.profile_baseline_samples > 0:
            profiler_cmd.extend(["--baseline-samples", str(args.profile_baseline_samples)])
        profiler_proc = subprocess.Popen(profiler_cmd)
        print(f"System profiler: {profile_output}")

    print("=== PATHBUNDLE CHOICE SCORER (V7, PUBLIC RELEASE) ===")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Device: {device}")
    print(f"Batches/epoch: {len(train_loader)} | Batch size: {args.batch_size}")
    print(
        f"Model: {args.model_family} | Scorer: {args.scorer_mode} | "
        f"Matcher: {args.matcher_mode}/{args.matcher_collapse} | "
        f"Norm: {args.norm_type} | Optimizer: {args.optimizer}"
    )

    use_tqdm = not args.no_tqdm
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, use_tqdm)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, None, device, use_tqdm)
            gap = val_acc - train_acc
            print(
                f"Epoch {epoch:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                f"Gap: {gap:+.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
    finally:
        if profiler_proc is not None:
            time.sleep(0.5)
            if profiler_proc.poll() is None:
                profiler_proc.terminate()


if __name__ == "__main__":
    main()
