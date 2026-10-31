"""Turn the LoRA telemetry pilots into a per-module risk table.

The pilots write one JSONL record per sampled optimizer step with per-module
gradient/update/parameter norms. This script aggregates those records into:

- a run-level quality check (records, steps, modules per record),
- per-module statistics (gradient level, volatility, update-to-weight ratio),
- cross-seed agreement for each module inside a model,
- a simple composite risk score used to rank modules.

It only reads artifacts; it never re-runs training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]

RUN_PATTERN = re.compile(r"telemetry_(?P<model>sailor2|qwen)[_-]?seed(?P<seed>\d+)", re.IGNORECASE)
MODEL_LABEL = {"sailor2": "Sailor2-1B-Chat", "qwen": "Qwen2.5-0.5B-Instruct"}
LOSS_PATTERN = re.compile(r"\{'loss': '([0-9.eE+-]+)'")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    if abs(average) < 1e-12:
        return 0.0
    return pstdev(values) / average


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average_rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = average_rank
        index = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    x_mean = mean(xs)
    y_mean = mean(values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator


def _module_kind(layer_key: str) -> str:
    tail = layer_key.rsplit(".", 1)[-1]
    for kind in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if tail == kind:
            return kind
    return tail


def _layer_index(layer_key: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", layer_key)
    return int(match.group(1)) if match else None


def _load_records(path: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    empty = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("layers"):
                empty += 1
                continue
            records.append(record)
    return records, empty


def _load_losses(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    losses: list[float] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOSS_PATTERN.search(line)
            if match:
                try:
                    losses.append(float(match.group(1)))
                except ValueError:
                    continue
    return losses


def _summarise_run(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"gradient_norm": [], "update_norm": [], "parameter_norm": [], "update_to_weight_ratio": []}
    )
    for record in records:
        for key, value in record["layers"].items():
            for metric in series[key]:
                series[key][metric].append(float(value.get(metric, 0.0)))
    return series


def _loss_spikes(losses: list[float], threshold: float = 3.5) -> dict[str, float]:
    if len(losses) < 4:
        return {"count": 0, "rate": 0.0, "max_z": 0.0}
    centre = median(losses)
    deviations = [abs(value - centre) for value in losses]
    mad = median(deviations)
    if mad < 1e-9:
        mad = max(mean(deviations), 1e-9)
    zscores = [0.6745 * (value - centre) / mad for value in losses]
    spikes = [z for z in zscores if abs(z) > threshold]
    return {
        "count": len(spikes),
        "rate": len(spikes) / len(losses),
        "max_z": max((abs(z) for z in zscores), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--pattern", default="telemetry_*.jsonl")
    parser.add_argument("--output-json", type=Path, default=ROOT / "results" / "telemetry_analysis.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "telemetry_analysis.md")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    run_files = sorted(args.results_dir.glob(args.pattern))
    if not run_files:
        raise SystemExit(f"No telemetry files match {args.pattern} in {args.results_dir}")

    runs: dict[str, dict] = {}
    for path in run_files:
        match = RUN_PATTERN.search(path.stem)
        if not match:
            print(f"skip (unrecognised name): {path.name}")
            continue
        model = match.group("model").lower()
        seed = int(match.group("seed"))
        records, empty = _load_records(path)
        if not records:
            print(f"skip (no usable records): {path.name}")
            continue
        losses = _load_losses(args.logs_dir / f"telemetry_{model}_seed{seed}" / "run.stdout.log")
        series = _summarise_run(records)
        runs[f"{model}_seed{seed}"] = {
            "model": model,
            "seed": seed,
            "records": len(records),
            "empty_records": empty,
            "steps": [int(record["step"]) for record in records],
            "modules": len(series),
            "total_gradient_norm": [float(record["total_gradient_norm"]) for record in records],
            "total_update_norm": [float(record["total_update_norm"]) for record in records],
            "update_to_weight_ratio": [float(record["update_to_weight_ratio"]) for record in records],
            "series": series,
            "losses": losses,
            "loss_spikes": _loss_spikes(losses),
        }

    if not runs:
        raise SystemExit("No run produced usable telemetry.")

    models: dict[str, list[dict]] = defaultdict(list)
    for run in runs.values():
        models[run["model"]].append(run)

    report: dict = {
        "runs": {},
        "models": {},
    }
    for name, run in sorted(runs.items()):
        report["runs"][name] = {
            "records": run["records"],
            "empty_records": run["empty_records"],
            "first_step": run["steps"][0],
            "last_step": run["steps"][-1],
            "modules": run["modules"],
            "total_gradient_norm_mean": mean(run["total_gradient_norm"]),
            "total_gradient_norm_max": max(run["total_gradient_norm"]),
            "update_to_weight_ratio_mean": mean(run["update_to_weight_ratio"]),
            "logged_losses": len(run["losses"]),
            "loss_spikes": run["loss_spikes"],
        }

    for model, model_runs in models.items():
        seeds = sorted(run["seed"] for run in model_runs)
        module_keys = sorted({key for run in model_runs for key in run["series"]})
        per_module: dict[str, dict] = {}
        for key in module_keys:
            per_seed: dict[str, dict[str, float]] = {}
            for run in sorted(model_runs, key=lambda item: item["seed"]):
                values = run["series"].get(key)
                if not values or not values["gradient_norm"]:
                    continue
                gradients = values["gradient_norm"]
                per_seed[str(run["seed"])] = {
                    "gradient_norm_mean": mean(gradients),
                    "gradient_norm_max": max(gradients),
                    "gradient_norm_cv": _cv(gradients),
                    "gradient_norm_slope": _slope(gradients) / (mean(gradients) + 1e-12),
                    "update_to_weight_ratio_mean": mean(values["update_to_weight_ratio"]),
                    "update_to_weight_ratio_max": max(values["update_to_weight_ratio"]),
                    "parameter_norm_last": values["parameter_norm"][-1],
                }
            if not per_seed:
                continue
            grad_means = [entry["gradient_norm_mean"] for entry in per_seed.values()]
            grad_cvs = [entry["gradient_norm_cv"] for entry in per_seed.values()]
            ratio_means = [entry["update_to_weight_ratio_mean"] for entry in per_seed.values()]
            per_module[key] = {
                "kind": _module_kind(key),
                "layer_index": _layer_index(key),
                "seeds": sorted(int(seed) for seed in per_seed),
                "per_seed": per_seed,
                "gradient_norm_mean": mean(grad_means),
                "gradient_norm_seed_sd": pstdev(grad_means) if len(grad_means) > 1 else 0.0,
                "gradient_norm_seed_cv": _cv(grad_means),
                "gradient_norm_within_run_cv": mean(grad_cvs),
                "update_to_weight_ratio_mean": mean(ratio_means),
                "update_to_weight_ratio_seed_cv": _cv(ratio_means),
            }

        # Rank each observable and combine equal-weight percentiles into a
        # single risk score. The score is a diagnostic ordering, not a
        # calibrated probability.
        keys = sorted(per_module)
        grad_levels = [per_module[key]["gradient_norm_mean"] for key in keys]
        grad_volatility = [per_module[key]["gradient_norm_within_run_cv"] for key in keys]
        grad_seed_spread = [per_module[key]["gradient_norm_seed_cv"] for key in keys]
        update_ratios = [per_module[key]["update_to_weight_ratio_mean"] for key in keys]

        def percentile_ranks(values: list[float]) -> list[float]:
            ranks = _ranks(values)
            if len(ranks) < 2:
                return [1.0] * len(ranks)
            return [(rank - 1) / (len(ranks) - 1) for rank in ranks]

        grad_rank = percentile_ranks(grad_levels)
        volatility_rank = percentile_ranks(grad_volatility)
        seed_rank = percentile_ranks(grad_seed_spread)
        update_rank = percentile_ranks(update_ratios)
        for index, key in enumerate(keys):
            per_module[key]["risk_score"] = mean(
                [grad_rank[index], volatility_rank[index], seed_rank[index], update_rank[index]]
            )

        seed_pairs = []
        if len(seeds) > 1:
            for first in range(len(seeds)):
                for second in range(first + 1, len(seeds)):
                    left_seed = str(seeds[first])
                    right_seed = str(seeds[second])
                    common = [
                        key
                        for key in keys
                        if left_seed in per_module[key]["per_seed"] and right_seed in per_module[key]["per_seed"]
                    ]
                    left = [per_module[key]["per_seed"][left_seed]["gradient_norm_mean"] for key in common]
                    right = [per_module[key]["per_seed"][right_seed]["gradient_norm_mean"] for key in common]
                    seed_pairs.append(
                        {
                            "seeds": [seeds[first], seeds[second]],
                            "modules": len(common),
                            "spearman_gradient_norm": _spearman(left, right),
                        }
                    )

        kind_values: dict[str, list[float]] = defaultdict(list)
        for key in keys:
            kind_values[per_module[key]["kind"]].append(per_module[key]["gradient_norm_mean"])
        kind_summary = {
            kind: {
                "modules": len(values),
                "gradient_norm_mean": mean(values),
                "risk_score_mean": mean(
                    [per_module[key]["risk_score"] for key in keys if per_module[key]["kind"] == kind]
                ),
            }
            for kind, values in sorted(kind_values.items())
        }

        layer_indices = [per_module[key]["layer_index"] for key in keys if per_module[key]["layer_index"] is not None]
        depth_summary = []
        if layer_indices:
            low, high = min(layer_indices), max(layer_indices)
            span = max(1, high - low + 1)
            buckets = 4
            for bucket in range(buckets):
                start = low + round(span * bucket / buckets)
                end = low + round(span * (bucket + 1) / buckets) - 1
                selected = [
                    per_module[key]
                    for key in keys
                    if per_module[key]["layer_index"] is not None and start <= per_module[key]["layer_index"] <= end
                ]
                if not selected:
                    continue
                depth_summary.append(
                    {
                        "layers": [start, end],
                        "modules": len(selected),
                        "gradient_norm_mean": mean(entry["gradient_norm_mean"] for entry in selected),
                        "risk_score_mean": mean(entry["risk_score"] for entry in selected),
                    }
                )

        ranking = sorted(keys, key=lambda key: per_module[key]["risk_score"], reverse=True)
        report["models"][model] = {
            "label": MODEL_LABEL.get(model, model),
            "seeds": seeds,
            "runs": sorted(run["seed"] for run in model_runs),
            "modules": len(keys),
            "seed_agreement": seed_pairs,
            "modules_summary": per_module,
            "module_kind_summary": kind_summary,
            "depth_summary": depth_summary,
            "top_risk_modules": [
                {"module": key, **{k: per_module[key][k] for k in (
                    "kind",
                    "layer_index",
                    "gradient_norm_mean",
                    "gradient_norm_within_run_cv",
                    "gradient_norm_seed_cv",
                    "update_to_weight_ratio_mean",
                    "risk_score",
                )}}
                for key in ranking[: args.top]
            ],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Telemetry analysis", ""]
    lines.append("Sinh tự động bởi `code/analyze_telemetry.py`; chỉ đọc artifact, không chạy lại training.")
    lines.append("")
    lines.append("## Chất lượng pilot")
    lines.append("")
    lines.append("| Run | Records | Empty | Steps | Modules | |g| mean | |g| max | update/weight mean | Loss spikes |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for name, run in sorted(report["runs"].items()):
        lines.append(
            "| {name} | {records} | {empty} | {first}–{last} | {modules} | {gmean:.4f} | {gmax:.4f} | {ratio:.2e} | {spikes} ({rate:.1%}) |".format(
                name=name,
                records=run["records"],
                empty=run["empty_records"],
                first=run["first_step"],
                last=run["last_step"],
                modules=run["modules"],
                gmean=run["total_gradient_norm_mean"],
                gmax=run["total_gradient_norm_max"],
                ratio=run["update_to_weight_ratio_mean"],
                spikes=run["loss_spikes"]["count"],
                rate=run["loss_spikes"]["rate"],
            )
        )
    lines.append("")
    for model, summary in report["models"].items():
        lines.append(f"## {summary['label']} (seeds {', '.join(str(seed) for seed in summary['seeds'])})")
        lines.append("")
        if summary["seed_agreement"]:
            agreement = ", ".join(
                f"{pair['seeds'][0]}↔{pair['seeds'][1]}: ρ={pair['spearman_gradient_norm']:.3f}"
                for pair in summary["seed_agreement"]
            )
            lines.append(f"- Cross-seed rank agreement of module gradient norms: {agreement}.")
        lines.append(f"- Modules observed: {summary['modules']}.")
        lines.append("")
        lines.append("| Module | Depth | |g| mean | within-run CV | cross-seed CV | update/weight | risk |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for entry in summary["top_risk_modules"]:
            depth = "-" if entry["layer_index"] is None else str(entry["layer_index"])
            lines.append(
                "| `{module}` | {depth} | {grad:.4f} | {cv:.3f} | {seed_cv:.3f} | {ratio:.2e} | {risk:.3f} |".format(
                    module=entry["module"],
                    depth=depth,
                    grad=entry["gradient_norm_mean"],
                    cv=entry["gradient_norm_within_run_cv"],
                    seed_cv=entry["gradient_norm_seed_cv"],
                    ratio=entry["update_to_weight_ratio_mean"],
                    risk=entry["risk_score"],
                )
            )
        lines.append("")
        lines.append("Module-kind aggregate:")
        lines.append("")
        lines.append("| Kind | Modules | |g| mean | risk mean |")
        lines.append("|---|---:|---:|---:|")
        for kind, entry in summary["module_kind_summary"].items():
            lines.append(
                f"| {kind} | {entry['modules']} | {entry['gradient_norm_mean']:.4f} | {entry['risk_score_mean']:.3f} |"
            )
        lines.append("")
        if summary["depth_summary"]:
            lines.append("Depth buckets:")
            lines.append("")
            lines.append("| Layers | Modules | |g| mean | risk mean |")
            lines.append("|---|---:|---:|---:|")
            for entry in summary["depth_summary"]:
                lines.append(
                    "| {start}–{end} | {modules} | {grad:.4f} | {risk:.3f} |".format(
                        start=entry["layers"][0],
                        end=entry["layers"][1],
                        modules=entry["modules"],
                        grad=entry["gradient_norm_mean"],
                        risk=entry["risk_score_mean"],
                    )
                )
            lines.append("")
    lines.append("`risk_score` is the equal-weight mean of the within-model percentile ranks of module gradient level, within-run gradient volatility, cross-seed gradient spread and update-to-weight ratio. It orders modules for follow-up; it is not a calibrated failure probability.")
    lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"runs: {len(runs)}  models: {len(report['models'])}")
    for model, summary in report["models"].items():
        print(f"  {model}: {summary['modules']} modules, {len(summary['top_risk_modules'])} ranked")
    print(f"wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
