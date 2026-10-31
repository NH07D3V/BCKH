"""Detect instability events in the long telemetry pilots.

Implements the label definition frozen in ``docs/gate2b_protocol.md``:

- warm-up window: records at steps 10-100,
- monitoring window: records at steps 110-1000,
- a spike is a monitoring record with robust z >= 4 and value >= 2x the
  warm-up median,
- a module is unstable when at least three distinct records spike.

The script writes ``results/instability_analysis.json`` and
``docs/instability_analysis.md``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from analyze_telemetry import MODEL_LABEL, RUN_PATTERN, _layer_index, _load_records, _module_kind

ROOT = Path(__file__).resolve().parents[1]
Z_THRESHOLD = 4.0
SPIKE_RECORDS = 3
WARMUP_LAST_STEP = 100
LOSS_PATTERN = re.compile(r"\{'loss': '([0-9.eE+-]+)'")


def _scaled_mad(values: list[float]) -> float:
    centre = median(values)
    deviations = [abs(value - centre) for value in values]
    mad = median(deviations)
    if mad > 0:
        return 1.4826 * mad
    if len(values) > 1:
        spread = float(np.std(values))
        if spread > 0:
            return spread
    return 0.0


def _module_events(warm_values: list[float], monitor_values: list[float]) -> dict:
    centre = median(warm_values)
    scale = _scaled_mad(warm_values)
    if scale <= 0 or centre <= 0:
        return {"valid": False, "spike_records": 0, "max_z": 0.0, "unstable": False}
    zscores = [(value - centre) / scale for value in monitor_values]
    spikes = [
        1
        for value, z in zip(monitor_values, zscores)
        if z >= Z_THRESHOLD and value >= 2.0 * centre
    ]
    return {
        "valid": True,
        "warmup_median": centre,
        "warmup_scale": scale,
        "spike_records": len(spikes),
        "max_z": max(zscores) if zscores else 0.0,
        "unstable": len(spikes) >= SPIKE_RECORDS,
    }


def _loss_spikes(log_path: Path) -> dict:
    if not log_path.exists():
        return {"logged": 0, "spikes": 0, "max_z": 0.0}
    losses = [
        float(match.group(1))
        for match in LOSS_PATTERN.finditer(log_path.read_text(encoding="utf-8", errors="replace"))
    ]
    if len(losses) < 4:
        return {"logged": len(losses), "spikes": 0, "max_z": 0.0}
    centre = median(losses)
    scale = _scaled_mad(losses)
    if scale <= 0:
        return {"logged": len(losses), "spikes": 0, "max_z": 0.0}
    zscores = [(value - centre) / scale for value in losses]
    spikes = sum(1 for z in zscores if z >= 4.0)
    return {"logged": len(losses), "spikes": spikes, "max_z": max(zscores)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--pattern", default="telemetry_long_*.jsonl")
    parser.add_argument("--output-json", type=Path, default=ROOT / "results" / "instability_analysis.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "instability_analysis.md")
    args = parser.parse_args()

    files = sorted(args.results_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match {args.pattern}")

    runs: dict[str, dict] = {}
    for path in files:
        match = RUN_PATTERN.search(path.stem)
        if not match:
            print(f"skip {path.name}")
            continue
        model = match.group("model").lower()
        seed = int(match.group("seed"))
        records, empty = _load_records(path)
        if not records:
            print(f"skip {path.name}: no usable records")
            continue
        warm = [record for record in records if int(record["step"]) <= WARMUP_LAST_STEP]
        monitor = [record for record in records if int(record["step"]) > WARMUP_LAST_STEP]
        if len(warm) < 3 or len(monitor) < 3:
            print(f"skip {path.name}: warm-up {len(warm)}, monitoring {len(monitor)}")
            continue
        modules = sorted(set(warm[0]["layers"]) & set(monitor[0]["layers"]))
        events: dict[str, dict] = {}
        for key in modules:
            warm_values = [float(record["layers"][key]["gradient_norm"]) for record in warm]
            monitor_values = [float(record["layers"][key]["gradient_norm"]) for record in monitor]
            events[key] = _module_events(warm_values, monitor_values)
        valid = {key: value for key, value in events.items() if value["valid"]}
        unstable = [key for key, value in valid.items() if value["unstable"]]
        loss = _loss_spikes(args.logs_dir / f"telemetry_long_{model}_seed{seed}" / "run.stdout.log")
        runs[f"{model}_seed{seed}"] = {
            "model": model,
            "seed": seed,
            "records": len(records),
            "first_step": int(records[0]["step"]),
            "last_step": int(records[-1]["step"]),
            "empty_records": empty,
            "modules": len(modules),
            "valid_modules": len(valid),
            "unstable_modules": len(unstable),
            "prevalence": len(unstable) / len(valid) if valid else 0.0,
            "spike_histogram": {
                str(count): sum(1 for value in valid.values() if value["spike_records"] == count)
                for count in sorted({value["spike_records"] for value in valid.values()})
            },
            "max_z": max((value["max_z"] for value in valid.values()), default=0.0),
            "loss": loss,
            "events": events,
        }

    report: dict = {"parameters": {"z_threshold": Z_THRESHOLD, "spike_records": SPIKE_RECORDS,
                                   "warmup_last_step": WARMUP_LAST_STEP},
                    "runs": {}, "models": {}}

    for name, run in sorted(runs.items()):
        report["runs"][name] = {key: value for key, value in run.items() if key != "events"}

    by_model: dict[str, list[dict]] = defaultdict(list)
    for run in runs.values():
        by_model[run["model"]].append(run)

    for model, model_runs in by_model.items():
        unstable_by_run = {}
        for run in model_runs:
            unstable_by_run[run["seed"]] = {
                key for key, value in run["events"].items()
                if value["valid"] and value["unstable"]
            }
        pair_agreement = {}
        seeds = sorted(unstable_by_run)
        for first in range(len(seeds)):
            for second in range(first + 1, len(seeds)):
                left = unstable_by_run[seeds[first]]
                right = unstable_by_run[seeds[second]]
                union = left | right
                pair_agreement[f"{seeds[first]}_{seeds[second]}"] = {
                    "shared": len(left & right),
                    "jaccard": len(left & right) / len(union) if union else 0.0,
                }
        aggregate: dict[str, dict] = {}
        keys = sorted({key for run in model_runs for key, value in run["events"].items() if value["valid"]})
        for key in keys:
            per_seed = {
                str(run["seed"]): run["events"][key]
                for run in model_runs
                if key in run["events"] and run["events"][key]["valid"]
            }
            aggregate[key] = {
                "kind": _module_kind(key),
                "layer_index": _layer_index(key),
                "seeds": sorted(int(seed) for seed in per_seed),
                "unstable_seeds": sorted(int(seed) for seed, value in per_seed.items() if value["unstable"]),
                "spike_records": {seed: value["spike_records"] for seed, value in per_seed.items()},
                "max_z": max(value["max_z"] for value in per_seed.values()),
            }

        kind_stats: dict[str, dict] = {}
        for key, entry in aggregate.items():
            bucket = kind_stats.setdefault(entry["kind"], {"modules": 0, "unstable": 0})
            bucket["modules"] += 1
            if entry["unstable_seeds"]:
                bucket["unstable"] += 1
        for bucket in kind_stats.values():
            bucket["rate"] = bucket["unstable"] / bucket["modules"] if bucket["modules"] else 0.0

        depth_stats = []
        depth_values = [entry["layer_index"] for entry in aggregate.values() if entry["layer_index"] is not None]
        if depth_values:
            low, high = min(depth_values), max(depth_values)
            span = max(1, high - low + 1)
            for bucket in range(4):
                start = low + round(span * bucket / 4)
                end = low + round(span * (bucket + 1) / 4) - 1
                selected = [
                    entry for entry in aggregate.values()
                    if entry["layer_index"] is not None and start <= entry["layer_index"] <= end
                ]
                if not selected:
                    continue
                depth_stats.append({
                    "layers": [start, end],
                    "modules": len(selected),
                    "unstable": sum(1 for entry in selected if entry["unstable_seeds"]),
                    "rate": sum(1 for entry in selected if entry["unstable_seeds"]) / len(selected),
                })

        report["models"][model] = {
            "label": MODEL_LABEL.get(model, model),
            "seeds": seeds,
            "prevalence": {
                str(run["seed"]): run["prevalence"] for run in model_runs
            },
            "unstable_per_run": {
                str(run["seed"]): run["unstable_modules"] for run in model_runs
            },
            "seed_agreement": pair_agreement,
            "module_kind": kind_stats,
            "depth": depth_stats,
            "modules": aggregate,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Instability analysis (long pilots)", ""]
    lines.append("Sinh tá»± Ä‘á»™ng bá»Ÿi `code/analyze_instability.py`. NhÃ£n theo `docs/gate2b_protocol.md`:")
    lines.append("spike khi robust z â‰¥ 4 vÃ  giÃ¡ trá»‹ â‰¥ 2Ã— median warm-up; module báº¥t á»•n Ä‘á»‹nh khi cÃ³ â‰¥ 3 record spike.")
    lines.append("")
    lines.append("## Tá»«ng run")
    lines.append("")
    lines.append("| Run | Records | Modules | Valid | Unstable | Prevalence | Max z | Loss spikes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, run in sorted(report["runs"].items()):
        lines.append(
            "| {name} | {records} | {modules} | {valid} | {unstable} | {prev:.1%} | {maxz:.1f} | {loss} |".format(
                name=name,
                records=run["records"],
                modules=run["modules"],
                valid=run["valid_modules"],
                unstable=run["unstable_modules"],
                prev=run["prevalence"],
                maxz=run["max_z"],
                loss=f"{run['loss']['spikes']}/{run['loss']['logged']}",
            )
        )
    lines.append("")
    for model, summary in report["models"].items():
        lines.append(f"## {summary['label']}")
        lines.append("")
        lines.append(
            "- Prevalence theo seed: "
            + ", ".join(f"{seed}: {value:.1%}" for seed, value in summary["prevalence"].items())
            + "."
        )
        if summary["seed_agreement"]:
            lines.append(
                "- Äá»“ng thuáº­n táº­p báº¥t á»•n Ä‘á»‹nh giá»¯a seed (Jaccard): "
                + ", ".join(
                    f"{pair}: {entry['jaccard']:.2f} ({entry['shared']} module chung)"
                    for pair, entry in summary["seed_agreement"].items()
                )
                + "."
            )
        lines.append("")
        lines.append("| Kind | Modules | Unstable | Rate |")
        lines.append("|---|---:|---:|---:|")
        for kind, entry in sorted(summary["module_kind"].items(), key=lambda item: item[1]["rate"], reverse=True):
            lines.append(f"| {kind} | {entry['modules']} | {entry['unstable']} | {entry['rate']:.1%} |")
        lines.append("")
        if summary["depth"]:
            lines.append("| Layers | Modules | Unstable | Rate |")
            lines.append("|---|---:|---:|---:|")
            for entry in summary["depth"]:
                lines.append(
                    f"| {entry['layers'][0]}â€“{entry['layers'][1]} | {entry['modules']} | "
                    f"{entry['unstable']} | {entry['rate']:.1%} |"
                )
            lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"runs: {len(report['runs'])}")
    for name, run in sorted(report["runs"].items()):
        print(f"  {name}: {run['unstable_modules']}/{run['valid_modules']} unstable ({run['prevalence']:.1%}), "
              f"loss spikes {run['loss']['spikes']}/{run['loss']['logged']}")
    print(f"wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()

