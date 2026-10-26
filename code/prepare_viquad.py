"""Normalize UIT-ViQuAD2.0 into JSONL while preserving split boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = Path(__file__).resolve().parents[1]


def _first_answer(example: dict[str, Any]) -> tuple[str, int | None]:
    answers = example.get("answers") or {}
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        return (str(texts[0]) if texts else "", int(starts[0]) if starts else None)
    if isinstance(answers, list) and answers:
        item = answers[0] or {}
        return str(item.get("text", "")), item.get("answer_start")
    return "", None


def _normalize(example: dict[str, Any], split: str) -> dict[str, Any]:
    # The released HF test snapshot has no gold `answers` field. Keep it for
    # inference, but mark it explicitly so no metric script treats it as a
    # negative example or a scored empty answer.
    answers_present = example.get("answers") is not None
    has_gold_label = answers_present or bool(example.get("is_impossible", False))
    answer, start = _first_answer(example)
    impossible = bool(example.get("is_impossible", False)) if has_gold_label else False
    context = str(example.get("context", ""))
    offset_valid: bool | None = None
    offset_aligned = start
    offset_repaired = False
    if has_gold_label and not impossible and answer and start is not None:
        offset_valid = context[start : start + len(answer)] == answer
        if not offset_valid:
            candidate = context.find(answer)
            if candidate >= 0:
                offset_aligned = candidate
                offset_repaired = True
    return {
        "id": str(example.get("id", "")),
        "title": str(example.get("title", "")),
        "context": context,
        "question": str(example.get("question", "")),
        "answer": "" if impossible else answer,
        "answer_start": start,
        "answer_start_aligned": offset_aligned,
        "answer_offset_valid": offset_valid,
        "answer_offset_repaired": offset_repaired,
        "is_impossible": impossible,
        "has_gold_label": has_gold_label,
        "source_split": split,
    }


def _write_split(dataset: Dataset, split: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"
    impossible = 0
    unlabeled = 0
    offset_invalid = 0
    offset_repaired = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            row = _normalize(example, split)
            impossible += int(row["is_impossible"])
            unlabeled += int(not row["has_gold_label"])
            offset_invalid += int(row["answer_offset_valid"] is False)
            offset_repaired += int(row["answer_offset_repaired"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "rows": len(dataset),
        "labeled": len(dataset) - unlabeled,
        "unlabeled": unlabeled,
        "unanswerable": impossible,
        "offset_invalid": offset_invalid,
        "offset_repaired": offset_repaired,
        "path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/raw/UIT-ViQuAD2.0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--create-dev", action="store_true", help="Make a deterministic dev split only if validation is absent.")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    loaded = load_from_disk(str(args.source_dir))
    if isinstance(loaded, DatasetDict):
        splits = dict(loaded)
    else:
        splits = {"train": loaded}

    if args.create_dev and "validation" not in splits and "train" in splits:
        split = splits["train"].train_test_split(test_size=0.1, seed=args.seed)
        splits["train"], splits["validation"] = split["train"], split["test"]

    summary = {name: _write_split(ds, name, args.output_dir) for name, ds in splits.items()}
    metadata_path = ROOT / "data" / "splits" / "viquad_split_manifest.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({"seed": args.seed, "splits": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

