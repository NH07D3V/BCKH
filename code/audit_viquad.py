"""Audit labels, answer offsets and context leakage across released splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(split: str) -> list[dict]:
    path = ROOT / "data/processed" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(row: dict) -> str:
    raw = "\x1f".join((row.get("title") or "", row.get("context") or ""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> None:
    names = ["train", "validation", "test"]
    rows = {name: _rows(name) for name in names}
    context_keys = {name: {_key(row) for row in rows[name]} for name in names}
    question_keys = {name: {row.get("question", "") for row in rows[name]} for name in names}
    summary = {}
    for name in names:
        summary[name] = {
            "rows": len(rows[name]),
            "labeled": sum(bool(row.get("has_gold_label", True)) for row in rows[name]),
            "unanswerable": sum(bool(row.get("is_impossible", False)) for row in rows[name]),
            "offset_invalid": sum(row.get("answer_offset_valid") is False for row in rows[name]),
            "offset_repaired": sum(bool(row.get("answer_offset_repaired", False)) for row in rows[name]),
        }
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}__{right}"] = {
            "context_title_overlap": len(context_keys[left] & context_keys[right]),
            "question_overlap": len(question_keys[left] & question_keys[right]),
        }
    result = {"summary": summary, "overlaps": overlaps}
    out = ROOT / "data/splits/viquad_quality_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

