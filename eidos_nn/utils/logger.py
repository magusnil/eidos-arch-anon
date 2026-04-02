import json
import platform
import socket
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class eidosLogger:
    """
    Eidos JSON Logger.
    Saves structured training logs (config, epoch metrics, timing) to JSON.
    """
    def __init__(self, run_name: str, config: Dict[str, Any], log_dir: str = "logs"):
        self.run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{run_name}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_path = self.log_dir / f"{self.run_id}.json"
        
        self.config = config
        self.start_time = time.time()
        
        self.history = {
            "identity": {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "model_name": config.get("model_name", "Unknown"),
                "params": config.get("params", 0)
            },
            "config": config,
            "reproducibility": self._build_reproducibility_record(config),
            "epochs": [],
            "timing": [],
            "outcome": {}
        }

    def _build_reproducibility_record(self, config: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "seed": config.get("seed"),
            "python_version": sys.version.split()[0],
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "hostname": socket.gethostname(),
            },
            "notes": (
                "Same numeric seed is not a cross-platform reproducibility guarantee. "
                "OS, device backend, kernel selection, and library version differences "
                "can change the realized run even when the seed matches."
            ),
        }

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            record["torch"] = {
                "version": torch.__version__,
                "initial_seed": int(torch.initial_seed()),
                "cuda_available": cuda_available,
                "num_threads": torch.get_num_threads(),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            }

            cudnn = getattr(torch.backends, "cudnn", None)
            if cudnn is not None:
                record["torch"]["cudnn"] = {
                    "enabled": bool(cudnn.enabled),
                    "benchmark": bool(cudnn.benchmark),
                    "deterministic": bool(cudnn.deterministic),
                }

            if cuda_available:
                record["torch"]["cuda"] = {
                    "version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                }
        except Exception as exc:
            record["torch"] = {"unavailable": True, "error": str(exc)}

        return record

    def log_progress(self, epoch: int, batch: int, total_batches: int, loss: float, acc: float):
        """Log progress every N batches to a text file for real-time monitoring."""
        progress_file = self.log_dir / f"{self.run_id}_progress.txt"
        with open(progress_file, "a") as f:
            f.write(f"Epoch {epoch} | Batch {batch}/{total_batches} | Loss: {loss:.4f} | Acc: {acc:.2f}%\n")

    def log_epoch(self, epoch_data: Dict[str, Any]):
        self.history["epochs"].append(epoch_data)
        self._save_json()

    def log_timing(self, epoch: int, timing_data: Dict[str, Any]):
        """
        Log timing data as a separate standardized channel.
        Keeps runtime/performance traces independent from task metrics.
        """
        entry = {"epoch": epoch}
        entry.update(timing_data or {})
        self.history["timing"].append(entry)
        self._save_json()

    def finalize(self, outcome: Dict[str, Any]):
        self.history["outcome"] = outcome
        self.history["outcome"]["wall_time"] = f"{time.time() - self.start_time:.2f}s"
        self._save_json()

    def _save_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2)
