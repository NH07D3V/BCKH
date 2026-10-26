"""Create a deterministic labeled holdout without splitting a context group."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(split: str) -> list[dict]:
    path = ROOT / "data/processed" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("has_gold_label", True)]


def _group_id(row: dict) -> str:
    raw = "\x1f".join((row.get("title") or "", row.get("context") or ""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-splits", nargs="+", default=["train"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/splits/grouped_holdout")
    args = parser.parse_args()
    if args.dev_fraction + args.test_fraction >= 1:
        raise ValueError("dev_fraction + test_fraction must be < 1")

    rows = [row for split in args.source_splits for row in _load(split)]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_id(row)].append(row)
    group_ids = list(groups)
    random.Random(args.seed).shuffle(group_ids)
    n_test = max(1, round(len(group_ids) * args.test_fraction))
    n_dev = max(1, round(len(group_ids) * args.dev_fraction))
    test_ids = set(group_ids[:n_test])
    dev_ids = set(group_ids[n_test : n_test + n_dev])

    buckets = {"train": [], "dev": [], "test": []}
    for group_id, group_rows in groups.items():
        bucket = "test" if group_id in test_ids else "dev" if group_id in dev_ids else "train"
        buckets[bucket].extend(group_rows)
    for bucket in buckets:
        buckets[bucket].sort(key=lambda row: str(row.get("id", "")))

    manifest = {
        "seed": args.seed,
        "source_splits": args.source_splits,
        "group_definition": "sha256(title + U+001F + context)",
        "fractions": {"dev": args.dev_fraction, "test": args.test_fraction},
        "groups": {"total": len(group_ids), "train": len(group_ids) - len(dev_ids) - len(test_ids), "dev": len(dev_ids), "test": len(test_ids)},
        "splits": {},
    }
    for bucket, bucket_rows in buckets.items():
        path = args.output_dir / f"{bucket}.jsonl"
        manifest["splits"][bucket] = {
            "rows": len(bucket_rows),
            "groups": len({ _group_id(row) for row in bucket_rows }),
            "sha256": _write(bucket_rows, path),
            "path": str(path),
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

