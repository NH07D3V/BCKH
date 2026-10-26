"""Verify the downloaded checkpoint and the labeled/unlabeled split contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA256 = {
    "Sailor2-1B-Chat": "3dd03235ed8fa190979856875c523d4e50b20577fa1b188f7075d6b8d22a09d7",
    "Qwen2.5-0.5B-Instruct": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
}
EXPECTED_PARQUET_SHA256 = {
    "train-00000-of-00001.parquet": "6e68a0c8fb6fa3c7f89e5b3825e3b3e96fd66913db89a9bdfcaf8fa2b5469e19",
    "validation-00000-of-00001.parquet": "8c81bdd75bd39eeff98df769f2282453397fa8c7004085e4fe3fc6e419733ae0",
    "test-00000-of-00001.parquet": "e6453e0fa0bc94128579b37a9ca4e8381c5301960944ca6399dccc85a785a805",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(split: str) -> list[dict]:
    path = ROOT / "data/processed" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    model_hashes = {}
    model_paths = {}
    for model_name, expected in EXPECTED_MODEL_SHA256.items():
        model_file = ROOT / "models/base" / model_name / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Missing {model_file}; run code/download_assets.py first."
            )
        actual_hash = sha256(model_file)
        if actual_hash != expected:
            raise RuntimeError(f"Model hash mismatch for {model_name}: {actual_hash}")
        model_hashes[model_name] = actual_hash
        model_paths[model_name] = str(model_file)
    parquet_hashes = {}
    parquet_root = ROOT / "data/raw/UIT-ViQuAD2.0_source/data"
    for filename, expected in EXPECTED_PARQUET_SHA256.items():
        actual = sha256(parquet_root / filename)
        if actual != expected:
            raise RuntimeError(f"Dataset hash mismatch for {filename}: {actual}")
        parquet_hashes[filename] = actual

    split_summary = {}
    for split in ("train", "validation", "test"):
        rows = read_rows(split)
        split_summary[split] = {
            "rows": len(rows),
            "labeled": sum(bool(row.get("has_gold_label", True)) for row in rows),
            "unanswerable": sum(bool(row.get("is_impossible", False)) for row in rows),
        }
    if split_summary["test"]["labeled"] != 0:
        raise RuntimeError("The released test snapshot unexpectedly contains labels; inspect before scoring.")

    result = {
        "models": {
            name: {"sha256": model_hashes[name], "path": model_paths[name]}
            for name in EXPECTED_MODEL_SHA256
        },
        # Backwards-compatible aliases for scripts that consumed the original
        # single-model verification result.
        "model_sha256": model_hashes["Qwen2.5-0.5B-Instruct"],
        "model_path": model_paths["Qwen2.5-0.5B-Instruct"],
        "parquet_sha256": parquet_hashes,
        "splits": split_summary,
        "status": "ok",
    }
    out = ROOT / "environment/asset_verification.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

