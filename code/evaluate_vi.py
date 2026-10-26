"""Evaluate prediction JSONL with Vietnamese-friendly SQuAD-style EM/F1."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction), normalize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = {}
    for token in pred:
        overlap[token] = min(pred.count(token), ref.count(token))
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="JSONL rows with prediction and reference/references fields")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for the metrics")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    em, score = [], []
    skipped_unlabeled = 0
    for row in rows:
        if row.get("has_gold_label", True) is False:
            skipped_unlabeled += 1
            continue
        refs = row.get("references", row.get("reference", row.get("answer", "")))
        if isinstance(refs, str):
            refs = [refs]
        pred = row.get("prediction", "")
        em.append(max(exact_match(pred, ref) for ref in refs))
        score.append(max(f1(pred, ref) for ref in refs))
    result = {
        "count": len(rows),
        "scored": len(em),
        "skipped_unlabeled": skipped_unlabeled,
        "exact_match": sum(em) / len(em) if em else 0.0,
        "f1": sum(score) / len(score) if score else 0.0,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

