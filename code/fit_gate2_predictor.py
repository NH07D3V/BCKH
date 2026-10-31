"""Gate 2 predictor experiment: does warm-up telemetry rank module risk?

Implements the protocol frozen in ``docs/gate2_protocol.md``:

- features come from the first five telemetry records (steps 10-50),
- labels come from the last five records (steps 160-200),
- train on two seeds, hold out the third, then transfer between models,
- ridge regression with the L2 strength picked by leave-one-seed-out,
- Spearman rho, bootstrap CI over modules, and a label-permutation test.

Numpy only: scikit-learn and scipy are not installed in this environment.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_telemetry import (
    MODEL_LABEL,
    RUN_PATTERN,
    _cv,
    _load_records,
    _module_kind,
    _layer_index,
    _slope,
    _spearman,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261031
BOOTSTRAP = 2000
PERMUTATIONS = 2000
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

CONTINUOUS = (
    "grad_mean_log",
    "grad_cv",
    "grad_slope",
    "update_ratio_mean",
    "update_ratio_max",
    "update_ratio_slope",
    "param_norm_log",
)
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

FEATURE_SETS = {
    "F_const": (),
    "F_loss": ("run_loss",),
    "F_grad": ("grad_mean_log", "grad_cv", "grad_slope"),
    "F_update": ("update_ratio_mean", "update_ratio_max", "update_ratio_slope"),
    "F_shape": ("depth_norm",) + tuple(f"kind_{kind}" for kind in KINDS),
    "F_full": CONTINUOUS
    + ("depth_norm",)
    + tuple(f"kind_{kind}" for kind in KINDS),
}
LABELS = ("late_update_ratio", "late_grad_cv")


def _rankdata(values: np.ndarray) -> np.ndarray:
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


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])


def ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    design = np.hstack([np.ones((features.shape[0], 1)), features])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def ridge_predict(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((features.shape[0], 1)), features]) @ coefficients


def _select_alpha(
    train_sets: list[tuple[np.ndarray, np.ndarray]], alphas: tuple[float, ...]
) -> float:
    """Pick the L2 strength that ranks the held-out training seed best."""
    if len(train_sets) < 2:
        return 1.0
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        scores = []
        for index, (features, target) in enumerate(train_sets):
            others = [item for position, item in enumerate(train_sets) if position != index]
            stacked_features = np.vstack([item[0] for item in others])
            stacked_target = np.concatenate([item[1] for item in others])
            coefficients = ridge_fit(stacked_features, stacked_target, alpha)
            prediction = ridge_predict(coefficients, features)
            scores.append(spearman(prediction, target))
        average = float(np.mean(scores))
        if average > best_score:
            best_alpha, best_score = alpha, average
    return best_alpha


def _build_rows(results_dir: Path) -> tuple[list[dict], dict]:
    files = sorted(results_dir.glob("telemetry_*.jsonl"))
    rows: list[dict] = []
    audit: dict[str, dict] = {}
    for path in files:
        match = RUN_PATTERN.search(path.stem)
        if not match:
            continue
        model = match.group("model").lower()
        seed = int(match.group("seed"))
        records, empty = _load_records(path)
        if not records:
            continue
        warm = records[:5]
        late = records[-5:]
        steps = [int(record["step"]) for record in records]
        if steps[0] > 10 or steps[-1] < 200:
            print(f"warning: {path.name} covers {steps[0]}â€“{steps[-1]}, expected 10â€“200")
        common = set(warm[0]["layers"])
        for record in warm + late:
            common &= set(record["layers"])
        dropped = len(warm[0]["layers"]) - len(common)
        total = int(warm[0]["layers"][next(iter(common))]["parameters"]) if common else 0
        run_loss = _run_loss(records)
        for key in sorted(common):
            gradient = [float(record["layers"][key]["gradient_norm"]) for record in warm]
            ratio = [float(record["layers"][key]["update_to_weight_ratio"]) for record in warm]
            parameter = [float(record["layers"][key]["parameter_norm"]) for record in warm]
            late_gradient = [float(record["layers"][key]["gradient_norm"]) for record in late]
            late_ratio = [float(record["layers"][key]["update_to_weight_ratio"]) for record in late]
            gradient_mean = float(np.mean(gradient))
            rows.append(
                {
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
                        "param_norm_log": float(np.log(float(np.mean(parameter)) + 1e-12)),
                        "run_loss": run_loss,
                    },
                    "labels": {
                        "late_update_ratio": float(np.mean(late_ratio)),
                        "late_grad_cv": _cv(late_gradient),
                    },
                    "depth": _layer_index(key),
                    "kind": _module_kind(key),
                }
            )
        audit[f"{model}_seed{seed}"] = {
            "records": len(records),
            "steps": [steps[0], steps[-1]],
            "modules_total": len(warm[0]["layers"]),
            "modules_used": len(common),
            "modules_dropped": dropped,
            "empty_records": empty,
            "parameter_elements_per_module": total,
        }
    return rows, audit


def _run_loss(records: list[dict]) -> float:
    values = [float(record.get("loss", 0.0)) for record in records if record.get("loss") is not None]
    return float(np.mean(values)) if values else 0.0


def _normalise(rows: list[dict]) -> None:
    """Z-score continuous features within each run, as the protocol requires."""
    per_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per_run[row["run"]].append(row)
    for run_rows in per_run.values():
        for feature in CONTINUOUS + ("depth_norm",):
            if feature == "depth_norm":
                depths = [row["depth"] for row in run_rows if row["depth"] is not None]
                if not depths:
                    continue
                low, high = min(depths), max(depths)
                span = max(1, high - low)
                for row in run_rows:
                    row["features"]["depth_norm"] = (
                        (row["depth"] - low) / span if row["depth"] is not None else 0.0
                    )
                continue
            values = np.array([row["features"][feature] for row in run_rows], dtype=float)
            spread = float(values.std())
            centre = float(values.mean())
            for row in run_rows:
                row["features"][feature] = (
                    (row["features"][feature] - centre) / spread if spread > 1e-12 else 0.0
                )
        for row in run_rows:
            for kind in KINDS:
                row["features"][f"kind_{kind}"] = 1.0 if row["kind"] == kind else 0.0


def _matrix(rows: list[dict], feature_set: str, label: str | None) -> tuple[np.ndarray, np.ndarray]:
    columns = FEATURE_SETS[feature_set]
    features = np.array([[row["features"][name] for name in columns] for row in rows], dtype=float)
    if features.ndim == 1:
        features = features.reshape(len(rows), len(columns)) if columns else np.zeros((len(rows), 0))
    target = np.array([row["labels"][label] for row in rows], dtype=float) if label else np.zeros(len(rows))
    return features, target


def _evaluate(
    train_rows: list[dict],
    test_rows: list[dict],
    feature_set: str,
    label: str,
    rng: np.random.Generator,
    bootstrap: int,
    permutations: int,
) -> dict:
    # alpha on the raw (already normalised) features, one model per training run
    per_seed: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        per_seed[row["run"]].append(row)
    train_sets = []
    for run_rows in per_seed.values():
        features, target = _matrix(run_rows, feature_set, label)
        if features.shape[1] == 0:
            features = np.zeros((len(run_rows), 1))
        train_sets.append((features, target))
    alpha = _select_alpha(train_sets, ALPHAS)
    # refit on the pooled training rows so the seeds are not averaged away
    pooled_features, pooled_target = _matrix(train_rows, feature_set, label)
    if pooled_features.shape[1] == 0:
        pooled_features = np.zeros((len(train_rows), 1))
    coefficients = ridge_fit(pooled_features, pooled_target, alpha)

    test_features, test_target = _matrix(test_rows, feature_set, label)
    if test_features.shape[1] == 0:
        test_features = np.zeros((len(test_rows), 1))
    prediction = ridge_predict(coefficients, test_features)
    rho = spearman(prediction, test_target)

    boot = np.empty(bootstrap)
    for index in range(bootstrap):
        sample = rng.integers(0, len(test_rows), len(test_rows))
        boot[index] = spearman(prediction[sample], test_target[sample])
    lower, upper = np.percentile(boot, [2.5, 97.5])

    null = np.empty(permutations)
    for index in range(permutations):
        shuffled = pooled_target.copy()
        rng.shuffle(shuffled)
        coefficients_null = ridge_fit(pooled_features, shuffled, alpha)
        null[index] = spearman(ridge_predict(coefficients_null, test_features), test_target)
    p_value = float((1 + np.sum(null >= rho)) / (permutations + 1))

    return {
        "rho": rho,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_permutation": p_value,
        "alpha": alpha,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "control_columns": FEATURE_SETS[feature_set],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gate2_predictor.json")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()


    rows, audit = _build_rows(args.results_dir)
    if not rows:
        raise SystemExit("No telemetry rows; run the pilot first.")
    _normalise(rows)

    by_model: dict[str, list[dict]] = defaultdict(list)
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
        by_run[row["run"]].append(row)

    splits: dict[str, dict] = {}
    honest_splits: dict[str, tuple[list[dict], list[dict]]] = {}
    for model, model_rows in by_model.items():
        seeds = sorted({row["seed"] for row in model_rows})
        if len(seeds) < 3:
            continue
        held_out = seeds[-1]
        split_name = f"{model}_seed{held_out}"
        honest_splits[split_name] = (
            [row for row in model_rows if row["seed"] != held_out],
            [row for row in model_rows if row["seed"] == held_out],
        )
    models = sorted(by_model)
    if len(models) == 2:
        honest_splits[f"{models[0]}_to_{models[1]}"] = (by_model[models[0]], by_model[models[1]])
        honest_splits[f"{models[1]}_to_{models[0]}"] = (by_model[models[1]], by_model[models[0]])

    rng = np.random.default_rng(SEED)
    for split_name, (train_rows, test_rows) in honest_splits.items():
        splits[split_name] = {"labels": {}, "rows": {"train": len(train_rows), "test": len(test_rows)}}
        for label in LABELS:
            splits[split_name]["labels"][label] = {}
            for feature_set in FEATURE_SETS:
                splits[split_name]["labels"][label][feature_set] = _evaluate(
                    train_rows,
                    test_rows,
                    feature_set,
                    label,
                    rng,
                    args.bootstrap,
                    args.permutations,
                )

    report = {
        "protocol": "docs/gate2_protocol.md",
        "seed": SEED,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "audit": audit,
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "labels": list(LABELS),
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for split_name, split in splits.items():
        print(f"\n=== {split_name} (train {split['rows']['train']}, test {split['rows']['test']}) ===")
        print(f"{'label':22s} {'feature':10s} {'rho':>7s} {'ci_low':>8s} {'p':>8s} {'delta':>7s}")
        for label, feature_sets in split["labels"].items():
            baseline = feature_sets["F_const"]["rho"]
            best_control = max(
                feature_sets[name]["rho"] for name in ("F_grad", "F_update", "F_shape", "F_loss")
            )
            for feature_set, result in feature_sets.items():
                delta = result["rho"] - (
                    best_control if feature_set == "F_full" else baseline
                )
                print(
                    f"{label:22s} {feature_set:10s} {result['rho']:7.3f} "
                    f"{result['ci_low']:8.3f} {result['p_permutation']:8.4f} {delta:7.3f}"
                )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()


