"""Shared deterministic parser for extractive QA generations."""

from __future__ import annotations

import re


def parse_prediction(raw: str) -> str:
    """Convert a chat-style continuation into the answer scored by EM/F1.

    The raw continuation is always saved separately. This parser is deliberately
    conservative: it only removes known explanation/answer markers and does not
    invent or rewrite the model's answer.
    """
    text = str(raw or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    no_answer = re.compile(r"^\s*\[KHÔNG CÓ ĐÁP ÁN\]\s*$", flags=re.IGNORECASE)
    if no_answer.match(text) or text.lower() in {"không có đáp án", "không có câu trả lời"}:
        return ""

    # If the model added an explanation after its answer, keep only the answer.
    for marker in ("### Giải thích", "### Explanation", "Write a short summary of the passage."):
        if marker.lower() in text.lower():
            index = text.lower().find(marker.lower())
            text = text[:index].strip()

    # If the model emitted an answer marker, keep the marked answer. When the
    # marker occurs after an earlier answer, the earlier segment is preferred.
    answer_markers = ("The answer is:", "Câu trả lời là:", "Câu trả lời:", "Đáp án:")
    for marker in answer_markers:
        index = text.lower().find(marker.lower())
        if index >= 0:
            prefix = text[:index].strip()
            suffix = text[index + len(marker):].strip()
            text = suffix if not prefix else prefix
            break

    # Generation should stop at a new prompt header if one appears.
    for marker in ("### Hướng dẫn", "### Đoạn văn", "### Câu hỏi", "### Trả lời"):
        index = text.lower().find(marker.lower())
        if index > 0:
            text = text[:index].strip()
            break
    return text
