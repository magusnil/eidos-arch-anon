"""
Lightweight system profiler for Eidos training runs.

Logs CPU, RAM, and NVIDIA GPU utilization to JSONL at a fixed interval.
It can optionally capture an idle baseline first and report deltas above
that baseline, which is usually a better estimate of true training cost
than raw desktop/IDE occupancy.

Examples:
    python public_release/profile_system_usage.py --interval 2.0 --output logs/system_profile.jsonl
    python public_release/profile_system_usage.py --baseline-samples 10 --output logs/system_profile.jsonl
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError as exc:
    raise SystemExit("psutil is required for profile_system_usage.py") from exc


def query_gpu():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_gpu_pct": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "memory_total_mb": float(parts[4]),
            }
        )
    return {"available": True, "gpus": gpus}


def add_baseline_delta(record, baseline):
    if baseline is None:
        return record

    baseline_memory = baseline["memory"]
    memory = record["memory"]
    memory["delta_used_mb"] = round(memory["used_mb"] - baseline_memory["used_mb"], 2)
    memory["delta_swap_used_mb"] = round(
        memory["swap_used_mb"] - baseline_memory["swap_used_mb"], 2
    )

    baseline_gpu = baseline.get("gpu", {})
    current_gpu = record.get("gpu", {})
    if baseline_gpu.get("available") and current_gpu.get("available"):
        baseline_by_index = {gpu["index"]: gpu for gpu in baseline_gpu["gpus"]}
        for gpu in current_gpu["gpus"]:
            base_gpu = baseline_by_index.get(gpu["index"])
            if base_gpu is None:
                continue
            gpu["delta_memory_used_mb"] = round(
                gpu["memory_used_mb"] - base_gpu["memory_used_mb"], 2
            )
            gpu["delta_utilization_gpu_pct"] = round(
                gpu["utilization_gpu_pct"] - base_gpu["utilization_gpu_pct"], 2
            )

    record["baseline_reference"] = {
        "timestamp": baseline["timestamp"],
        "mode": "idle_startup_average",
    }
    return record


def sample():
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_pct = psutil.cpu_percent(interval=None)

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "utilization_pct": cpu_pct,
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
        },
        "memory": {
            "total_mb": round(vm.total / (1024 * 1024), 2),
            "used_mb": round(vm.used / (1024 * 1024), 2),
            "available_mb": round(vm.available / (1024 * 1024), 2),
            "percent": vm.percent,
            "swap_used_mb": round(swap.used / (1024 * 1024), 2),
            "swap_percent": swap.percent,
        },
        "gpu": query_gpu(),
    }


def capture_baseline(sample_count, interval):
    if sample_count <= 0:
        return None

    print(
        f"Capturing idle baseline from {sample_count} samples "
        f"at {interval:.2f}s intervals before main logging."
    )

    samples = []
    for i in range(sample_count):
        samples.append(sample())
        if i < sample_count - 1:
            time.sleep(interval)

    baseline = json.loads(json.dumps(samples[-1]))
    baseline["timestamp"] = datetime.now().isoformat()

    baseline["cpu"]["utilization_pct"] = round(
        sum(s["cpu"]["utilization_pct"] for s in samples) / sample_count, 2
    )
    baseline["memory"]["used_mb"] = round(
        sum(s["memory"]["used_mb"] for s in samples) / sample_count, 2
    )
    baseline["memory"]["available_mb"] = round(
        sum(s["memory"]["available_mb"] for s in samples) / sample_count, 2
    )
    baseline["memory"]["percent"] = round(
        sum(s["memory"]["percent"] for s in samples) / sample_count, 2
    )
    baseline["memory"]["swap_used_mb"] = round(
        sum(s["memory"]["swap_used_mb"] for s in samples) / sample_count, 2
    )
    baseline["memory"]["swap_percent"] = round(
        sum(s["memory"]["swap_percent"] for s in samples) / sample_count, 2
    )

    if baseline["gpu"].get("available"):
        gpu_lists = [s["gpu"]["gpus"] for s in samples if s["gpu"].get("available")]
        if gpu_lists:
            gpu_count = len(gpu_lists[0])
            averaged = []
            for idx in range(gpu_count):
                first = gpu_lists[0][idx]
                averaged.append(
                    {
                        "index": first["index"],
                        "name": first["name"],
                        "utilization_gpu_pct": round(
                            sum(g[idx]["utilization_gpu_pct"] for g in gpu_lists)
                            / len(gpu_lists),
                            2,
                        ),
                        "memory_used_mb": round(
                            sum(g[idx]["memory_used_mb"] for g in gpu_lists) / len(gpu_lists),
                            2,
                        ),
                        "memory_total_mb": first["memory_total_mb"],
                    }
                )
            baseline["gpu"]["gpus"] = averaged

    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=0,
        help="Number of idle samples to average before logging deltas above baseline.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="system_profile",
        help="Tag used when auto-generating an output filename.",
    )
    parser.add_argument(
        "--stop-on-pid",
        type=int,
        default=0,
        help="Stop automatically when this PID no longer exists.",
    )
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path("logs") / f"{args.tag}_{timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = capture_baseline(args.baseline_samples, args.interval)

    print(f"Profiling to {output_path} every {args.interval:.2f}s. Press Ctrl+C to stop.")
    if baseline is not None:
        print("Baseline captured; output will include delta_* fields above idle.")
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "record_type": "baseline",
                        "baseline": baseline,
                    }
                )
                + "\n"
            )

    try:
        while True:
            if args.stop_on_pid and not psutil.pid_exists(args.stop_on_pid):
                print(f"Watched PID {args.stop_on_pid} exited; stopping profiler.")
                break
            record = sample()
            record = add_baseline_delta(record, baseline)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
