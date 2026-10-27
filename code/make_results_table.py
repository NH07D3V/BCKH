"""Assemble reproducible baseline summaries from run and prediction artifacts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("Sailor2-1B-Chat", "sailor2_seed13"),
    ("Sailor2-1B-Chat", "sailor2_seed37"),
    ("Sailor2-1B-Chat", "sailor2_seed73"),
    ("Qwen2.5-0.5B-Instruct", "qwen_seed13"),
    ("Qwen2.5-0.5B-Instruct", "qwen_seed37"),
    ("Qwen2.5-0.5B-Instruct", "qwen_seed73"),
]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for model, name in RUNS:
        run_metrics_path = ROOT / "checkpoints" / f"baseline_{name}" / "run_metrics.json"
        eval_metrics_path = ROOT / "results" / f"{name}_dev_metrics.json"
        if not run_metrics_path.exists() or not eval_metrics_path.exists():
            continue
        run_metrics = _read(run_metrics_path)
        eval_metrics = _read(eval_metrics_path)
        train_metrics = run_metrics.get("train_metrics", {})
        rows.append({
            "model": model,
            "run": name,
            "seed": run_metrics.get("seed"),
            "exact_match": eval_metrics.get("exact_match"),
            "f1": eval_metrics.get("f1"),
            "eval_count": eval_metrics.get("scored"),
            "train_loss": train_metrics.get("train_loss"),
            "wall_clock_seconds": run_metrics.get("wall_clock_seconds"),
            "peak_cuda_memory_allocated_bytes": run_metrics.get("peak_cuda_memory_allocated_bytes"),
            "peak_cuda_memory_reserved_bytes": run_metrics.get("peak_cuda_memory_reserved_bytes"),
        })

    summary = {"runs": rows, "groups": {}}
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        summary["groups"][model] = {
            "n": len(group),
            "exact_match_mean": statistics.mean(row["exact_match"] for row in group),
            "exact_match_sd": statistics.stdev(row["exact_match"] for row in group) if len(group) > 1 else None,
            "f1_mean": statistics.mean(row["f1"] for row in group),
            "f1_sd": statistics.stdev(row["f1"] for row in group) if len(group) > 1 else None,
        }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Baseline summary", "", "Generated from run_metrics and dev metric artifacts.", "", "| Model | Run | Seed | EM | F1 | Train loss | Runtime (s) | Peak VRAM (GiB) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        memory = row["peak_cuda_memory_allocated_bytes"]
        memory_gib = memory / (1024 ** 3) if memory is not None else None
        lines.append(f"| {row['model']} | {row['run']} | {row['seed']} | {row['exact_match']:.4f} | {row['f1']:.4f} | {row['train_loss']} | {row['wall_clock_seconds']} | {memory_gib:.2f} |" if memory_gib is not None else f"| {row['model']} | {row['run']} | {row['seed']} | {row['exact_match']:.4f} | {row['f1']:.4f} | {row['train_loss']} | {row['wall_clock_seconds']} | TBD |")
    lines.extend([
        "",
        "*Sailor2 seed 13 was resumed from checkpoint-250: runtime and train loss cover steps 250-2,837 only. EM/F1 and final eval loss are computed on the full grouped dev.*",
    ])
    (ROOT / "paper" / "baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
