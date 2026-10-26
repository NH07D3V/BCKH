"""Download the pinned research models and dataset.

The Vietnamese-adapted Sailor2 checkpoint is the primary model.  Qwen is
kept as a same-workspace control model so that any gain can be attributed to
the fine-tuning/optimization method rather than to the choice of checkpoint.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "taidng/UIT-ViQuAD2.0"
DATASET_REVISION = "406f09a45cc106a8f7b7fd0c25078883fe58cb1f"

MODEL_SPECS = (
    {
        "role": "primary",
        "repo_id": "sail/Sailor2-1B-Chat",
        "revision": "51b48ecd7c0629e4c79dc927a0445e6b671d8692",
        "directory": "Sailor2-1B-Chat",
    },
    {
        "role": "control",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "directory": "Qwen2.5-0.5B-Instruct",
    },
)


def main() -> None:
    dataset_dir = PROJECT_ROOT / "data" / "raw" / "UIT-ViQuAD2.0"
    dataset_source_dir = PROJECT_ROOT / "data" / "raw" / "UIT-ViQuAD2.0_source"
    models_root = PROJECT_ROOT / "models" / "base"
    models_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    model_manifests = []
    for spec in MODEL_SPECS:
        model_dir = models_root / spec["directory"]
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec['role']} model {spec['repo_id']} -> {model_dir}")
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="model",
            revision=spec["revision"],
            local_dir=str(model_dir),
        )
        model_manifests.append(
            {
                "role": spec["role"],
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "local_dir": str(model_dir),
            }
        )

    print(f"Downloading dataset {DATASET_ID}")
    # Preserve the original Parquet files and explicitly map only known splits;
    # do not execute repository-side processing code or auto-discover files.
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(dataset_source_dir),
        allow_patterns=["data/*.parquet"],
    )
    parquet_root = dataset_source_dir / "data"
    data_files = {
        split: str(parquet_root / f"{split}-00000-of-00001.parquet")
        for split in ("train", "validation", "test")
    }
    dataset = load_dataset("parquet", data_files=data_files)
    dataset.save_to_disk(str(dataset_dir))

    split_summary = {}
    if isinstance(dataset, DatasetDict):
        for split_name, split in dataset.items():
            split_summary[split_name] = {
                "rows": len(split),
                "columns": list(split.column_names),
                "fingerprint": getattr(split, "_fingerprint", None),
            }
    else:
        split_summary["train"] = {
            "rows": len(dataset),
            "columns": list(dataset.column_names),
            "fingerprint": getattr(dataset, "_fingerprint", None),
        }

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        # Keep ``model`` as a backwards-compatible alias for the primary
        # model; new consumers should use ``models`` and the explicit roles.
        "model": model_manifests[0],
        "primary_model": model_manifests[0],
        "control_model": model_manifests[1],
        "models": model_manifests,
        "dataset": {
            "repo_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "local_dir": str(dataset_dir),
            "source_parquet_dir": str(dataset_source_dir),
            "splits": split_summary,
        },
    }
    manifest_path = PROJECT_ROOT / "environment" / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

