"""Gate 2b: predict module instability from warm-up telemetry.

Follows ``docs/gate2b_protocol.md``:

- features from records at steps 10-100,
- binary label from ``results/instability_analysis.json``,
- ridge linear probability, L2 strength by leave-one-seed-out AUROC,
- AUROC / AUPRC with bootstrap CI and a label-permutation test,
- train seeds 13+37, hold out 73, then Sailor2<->Qwen transfer.

Artifact: ``results/gate2b_predictor.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_telemetry import MODEL_LABEL, RUN_PATTERN, _cv, _layer_index, _load_records, _module_kind, _slope

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261031
BOOTSTRAP = 2000
PERMUTATIONS = 2000
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
WARMUP_LAST_STEP = 100
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

CONTINUOUS = (
    "grad_mean_log",
    "grad_cv",
    "grad_slope",
    "update_ratio_mean",
    "update_ratio_max",
    "update_ratio_slope",
    "cosine_mean",
    "cosine_sd",
    "cosine_negative_fraction",
)

FEATURE_SETS = {
    "F_shape": ("depth_norm",) + tuple(f"kind_{kind}" for kind in KINDS),
    "F_grad": ("grad_mean_log", "grad_cv", "grad_slope"),
    "F_update": ("update_ratio_mean", "update_ratio_max", "update_ratio_slope"),
    "F_cosine": ("cosine_mean", "cosine_sd", "cosine_negative_fraction"),
    "F_full": CONTINUOUS + ("depth_norm",) + tuple(f"kind_{kind}" for kind in KINDS),
}


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2 + 1
        index = end + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    ranks = _ranks(scores)
    rank_sum = ranks[positive].sum()
    count_pos = int(positive.sum())
    count_neg = int(negative.sum())
    return float((rank_sum - count_pos * (count_pos + 1) / 2) / (count_pos * count_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels == 1
    if positive.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    hits = 0
    precision_sum = 0.0
    for position, label in enumerate(sorted_labels, start=1):
        if label == 1:
            hits += 1
            precision_sum += hits / position
    return float(precision_sum / positive.sum())


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    if left_ranks.std() < 1e-12 or right_ranks.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    design = np.hstack([np.ones((features.shape[0], 1)), features])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def ridge_predict(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((features.shape[0], 1)), features]) @ coefficients


def _select_alpha(folds: list[tuple[np.ndarray, np.ndarray]], alphas: tuple[float, ...]) -> float:
    if len(folds) < 2:
        return 1.0
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        scores = []
        for index, (features, target) in enumerate(folds):
            others = [item for position, item in enumerate(folds) if position != index]
            weights = ridge_fit(np.vstack([item[0] for item in others]),
                                np.concatenate([item[1] for item in others]), alpha)
            score = auroc(ridge_predict(weights, features), target)
            if not np.isnan(score):
                scores.append(score)
        if not scores:
            continue
        average = float(np.mean(scores))
        if average > best_score:
            best_alpha, best_score = alpha, average
    return best_alpha


def _build_rows(results_dir: Path, instability: dict) -> tuple[list[dict], dict]:
    labels: dict[tuple[str, int, str], dict] = {}
    for model, summary in instability.get("models", {}).items():
        for module, entry in summary.get("modules", {}).items():
            for seed in entry["seeds"]:
                labels[(model, int(seed), module)] = entry

    rows: list[dict] = []
    audit: dict[str, dict] = {}
    for path in sorted(results_dir.glob("telemetry_long_*.jsonl")):
        match = RUN_PATTERN.search(path.stem)
        if not match:
            continue
        model = match.group("model").lower()
        seed = int(match.group("seed"))
        records, empty = _load_records(path)
        if not records:
            continue
        warm = [record for record in records if int(record["step"]) <= WARMUP_LAST_STEP]
        if len(warm) < 5:
            print(f"skip {path.name}: warm-up has {len(warm)} records")
            continue
        used = 0
        for key in sorted(warm[0]["layers"]):
            entry = labels.get((model, seed, key))
            if entry is None:
                continue
            gradient = [float(record["layers"][key]["gradient_norm"]) for record in warm]
            ratio = [float(record["layers"][key]["update_to_weight_ratio"]) for record in warm]
            cosine = [float(record["layers"][key].get("gradient_cosine", 0.0)) for record in warm][1:]
            gradient_mean = float(np.mean(gradient))
            cosine_array = np.array(cosine) if cosine else np.array([0.0])
            rows.append({
                "run": f"{model}_seed{seed}",
                "model": model,
                "seed": seed,
                "module": key,
                "features": {
                    "grad_mean_log": float(np.log(gradient_mean + 1e-12)),
                    "grad_cv": _cv(gradient),
                    "grad_slope": _slope(gradient) / (gradient_mean + 1e-12),
                    "update_ratio_mean": float(np.mean(ratio)),
                    "update_ratio_max": float(np.max(ratio)),
                    "update_ratio_slope": _slope(ratio) / (float(np.mean(ratio)) + 1e-12),
                    "cosine_mean": float(cosine_array.mean()),
                    "cosine_sd": float(cosine_array.std()),
                    "cosine_negative_fraction": float(np.mean(cosine_array < 0)),
                },
                "depth": _layer_index(key),
                "kind": _module_kind(key),
                "label": 1.0 if seed in entry["unstable_seeds"] else 0.0,
                "spike_records": float(entry["spike_records"].get(str(seed), 0.0)),
            })
            used += 1
        audit[f"{model}_seed{seed}"] = {
            "records": len(records),
            "warmup_records": len(warm),
            "modules_used": used,
            "empty_records": empty,
        }
    return rows, audit


def _normalise(rows: list[dict]) -> None:
    per_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per_run[row["run"]].append(row)
    for run_rows in per_run.values():
        for feature in CONTINUOUS:
            values = np.array([row["features"][feature] for row in run_rows], dtype=float)
            centre, spread = float(values.mean()), float(values.std())
            for row in run_rows:
                row["features"][feature] = (
                    (row["features"][feature] - centre) / spread if spread > 1e-12 else 0.0
                )
        depths = [row["depth"] for row in run_rows if row["depth"] is not None]
        low, high = (min(depths), max(depths)) if depths else (0, 1)
        span = max(1, high - low)
        for row in run_rows:
            row["features"]["depth_norm"] = (row["depth"] - low) / span if row["depth"] is not None else 0.0
            for kind in KINDS:
                row["features"][f"kind_{kind}"] = 1.0 if row["kind"] == kind else 0.0


def _matrix(rows: list[dict], feature_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = FEATURE_SETS[feature_set]
    features = np.array([[row["features"][name] for name in columns] for row in rows], dtype=float)
    if features.ndim == 1:
        features = features.reshape(len(rows), len(columns))
    labels = np.array([row["label"] for row in rows], dtype=float)
    spikes = np.array([row["spike_records"] for row in rows], dtype=float)
    return features, labels, spikes


def _evaluate(train_rows: list[dict], test_rows: list[dict], feature_set: str,
              rng: np.random.Generator, bootstrap: int, permutations: int) -> dict:
    per_seed: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        per_seed[row["run"]].append(row)
    folds = []
    for run_rows in per_seed.values():
        features, labels, _ = _matrix(run_rows, feature_set)
        folds.append((features, labels))
    alpha = _select_alpha(folds, ALPHAS)

    train_features, train_labels, _ = _matrix(train_rows, feature_set)
    weights = ridge_fit(train_features, train_labels, alpha)
    test_features, test_labels, test_spikes = _matrix(test_rows, feature_set)
    scores = ridge_predict(weights, test_features)

    observed = auroc(scores, test_labels)
    average_precision = auprc(scores, test_labels)
    prevalence = float(test_labels.mean())

    boots = []
    for _ in range(bootstrap):
        sample = rng.integers(0, len(test_rows), len(test_rows))
        value = auroc(scores[sample], test_labels[sample])
        if not np.isnan(value):
            boots.append(value)
    ci_low, ci_high = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))

    null = np.empty(permutations)
    for index in range(permutations):
        shuffled = train_labels.copy()
        rng.shuffle(shuffled)
        weights_null = ridge_fit(train_features, shuffled, alpha)
        null[index] = auroc(ridge_predict(weights_null, test_features), test_labels)
    valid_null = null[~np.isnan(null)]
    p_value = float((1 + np.sum(valid_null >= observed)) / (len(valid_null) + 1)) if len(valid_null) else float("nan")

    return {
        "auroc": float(observed),
        "auprc": float(average_precision),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_permutation": p_value,
        "prevalence": prevalence,
        "spearman_spike_records": spearman(scores, test_spikes),
        "alpha": alpha,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--instability", type=Path, default=ROOT / "results" / "instability_analysis.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gate2b_predictor.json")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()

    instability = json.loads(args.instability.read_text(encoding="utf-8"))
    rows, audit = _build_rows(args.results_dir, instability)
    if not rows:
        raise SystemExit("No rows; run the long pilots and analyze_instability.py first.")
    _normalise(rows)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    splits: dict[str, tuple[list[dict], list[dict]]] = {}
    for model, model_rows in by_model.items():
        seeds = sorted({row["seed"] for row in model_rows})
        if len(seeds) < 3:
            continue
        held_out = seeds[-1]
        splits[f"{model}_seed{held_out}"] = (
            [row for row in model_rows if row["seed"] != held_out],
            [row for row in model_rows if row["seed"] == held_out],
        )
    models = sorted(by_model)
    if len(models) == 2:
        splits[f"{models[0]}_to_{models[1]}"] = (by_model[models[0]], by_model[models[1]])
        splits[f"{models[1]}_to_{models[0]}"] = (by_model[models[1]], by_model[models[0]])

    rng = np.random.default_rng(SEED)
    report: dict = {
        "protocol": "docs/gate2b_protocol.md",
        "seed": SEED,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "audit": audit,
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "splits": {},
    }

    for split_name, (train_rows, test_rows) in splits.items():
        report["splits"][split_name] = {
            "rows": {"train": len(train_rows), "test": len(test_rows)},
            "feature_sets": {},
        }
        for feature_set in FEATURE_SETS:
            report["splits"][split_name]["feature_sets"][feature_set] = _evaluate(
                train_rows, test_rows, feature_set, rng, args.bootstrap, args.permutations
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for split_name, split in report["splits"].items():
        print(f"\n=== {split_name} (train {split['rows']['train']}, test {split['rows']['test']}) ===")
        print(f"{'feature':10s} {'auroc':>7s} {'ci_low':>7s} {'auprc':>7s} {'p':>8s} {'rho_spikes':>10s}")
        for feature_set, result in split["feature_sets"].items():
            print(
                f"{feature_set:10s} {result['auroc']:7.3f} {result['ci_low']:7.3f} "
                f"{result['auprc']:7.3f} {result['p_permutation']:8.4f} "
                f"{result['spearman_spike_records']:10.3f}"
            )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
