"""Measure prompt/answer truncation for the frozen QA formatting protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompt(row: dict) -> str:
    answer = "[KHÔNG CÓ ĐÁP ÁN]" if row.get("is_impossible") else (row.get("answer") or "")
    return (
        "### Hướng dẫn\nTrả lời câu hỏi bằng thông tin trong đoạn văn.\n"
        f"### Đoạn văn\n{row.get('context', '')}\n"
        f"### Câu hỏi\n{row.get('question', '')}\n"
        f"### Trả lời\n{answer}"
    )


def _input_prompt(row: dict) -> str:
    return (
        "### Hướng dẫn\nTrả lời câu hỏi bằng thông tin trong đoạn văn.\n"
        f"### Đoạn văn\n{row.get('context', '')}\n"
        f"### Câu hỏi\n{row.get('question', '')}\n"
        "### Trả lời\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/grouped_holdout/train.jsonl")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model if args.model.is_absolute() else ROOT / args.model
    split = args.split if args.split.is_absolute() else ROOT / args.split
    rows = _rows(split)
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    records = []
    prompt_truncated = 0
    answer_truncated = 0
    total_tokens = 0
    answer_tokens = 0
    for row in rows:
        prompt_ids = tokenizer(_input_prompt(row), add_special_tokens=True, truncation=False)["input_ids"]
        full_ids = tokenizer(_prompt(row), add_special_tokens=True, truncation=False)["input_ids"]
        clipped = tokenizer(_prompt(row), add_special_tokens=True, truncation=True, max_length=args.max_length)["input_ids"]
        answer_ids = tokenizer(("[KHÔNG CÓ ĐÁP ÁN]" if row.get("is_impossible") else (row.get("answer") or "")), add_special_tokens=False)["input_ids"]
        total_tokens += len(full_ids)
        answer_tokens += len(answer_ids)
        prompt_would_truncate = len(prompt_ids) > args.max_length
        answer_would_truncate = len(full_ids) > args.max_length and len(clipped) <= args.max_length
        prompt_truncated += int(prompt_would_truncate)
        answer_truncated += int(answer_would_truncate)
        if len(records) < 5:
            records.append({
                "id": row.get("id"),
                "full_tokens": len(full_ids),
                "prompt_tokens": len(prompt_ids),
                "answer_tokens": len(answer_ids),
                "clipped_tokens": len(clipped),
                "prompt_exceeds_limit": prompt_would_truncate,
            })
    result = {
        "model_dir": str(model_dir),
        "split": str(split),
        "rows": len(rows),
        "max_length": args.max_length,
        "prompt_exceeds_limit": prompt_truncated,
        "prompt_exceeds_limit_fraction": prompt_truncated / len(rows) if rows else 0.0,
        "full_sequence_exceeds_limit": answer_truncated,
        "full_sequence_exceeds_limit_fraction": answer_truncated / len(rows) if rows else 0.0,
        "mean_full_tokens": total_tokens / len(rows) if rows else 0.0,
        "mean_answer_tokens": answer_tokens / len(rows) if rows else 0.0,
        "sha256_split": hashlib.sha256(split.read_bytes()).hexdigest(),
        "examples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
