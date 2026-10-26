"""Generate validation/test predictions from a saved LoRA adapter.

The default split is validation. Test must be requested explicitly so it is
harder to use the held-out split while choosing hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prediction_utils import parse_prediction


ROOT = Path(__file__).resolve().parents[1]


def _prompt(row: dict) -> str:
    return (
        "### Hướng dẫn\nTrả lời câu hỏi bằng thông tin trong đoạn văn.\n"
        f"### Đoạn văn\n{row.get('context', '')}\n"
        f"### Câu hỏi\n{row.get('question', '')}\n"
        "### Trả lời\n"
    )


def _find_run_config(adapter: Path) -> Path | None:
    """Find the config saved next to an adapter or in its parent output dir."""
    for candidate in (adapter / "run_config.json", adapter.parent / "run_config.json"):
        if candidate.exists():
            return candidate
    return None


def _resolve_base_dir(adapter: Path, explicit: Path | None) -> tuple[Path, dict]:
    """Resolve the base checkpoint without hard-coding the control model."""
    if explicit is not None:
        base_dir = explicit if explicit.is_absolute() else ROOT / explicit
        return base_dir, {}

    run_config_path = _find_run_config(adapter)
    if run_config_path is not None:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        local_dir = run_config.get("model", {}).get("local_dir")
        if local_dir:
            base_dir = Path(local_dir)
            if not base_dir.is_absolute():
                base_dir = ROOT / base_dir
            return base_dir, run_config

    # Backwards-compatible fallback for adapters created before model
    # inference was added.
    return ROOT / "models/base/Qwen2.5-0.5B-Instruct", {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=ROOT / "checkpoints/baseline_qlora_vi")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Optional local base checkpoint; otherwise infer it from run_config.json.",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--data-file", type=Path, default=None, help="Optional custom labeled JSONL, e.g. grouped_holdout/test.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size; use 1 if VRAM is tight.")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.split == "test":
        print("WARNING: test predictions are for the final frozen evaluation only.")
    base_dir, run_config = _resolve_base_dir(args.adapter, args.base_model)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base model not found: {base_dir}. Pass --base-model or download the pinned assets."
        )
    data_file = args.data_file or (ROOT / "data/processed" / f"{args.split}.jsonl")
    if not data_file.is_absolute():
        data_file = ROOT / data_file
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    rows = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer_dir = args.adapter if (args.adapter / "tokenizer.json").exists() else base_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_4bit = not args.no_4bit
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit generation requires CUDA; use --no-4bit only for a debug run.")
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    model.eval()
    device = next(model.parameters()).device

    output = args.output or (ROOT / "results" / f"predictions_{args.split}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        batch_size = max(1, int(args.batch_size))
        progress = tqdm(total=len(rows), desc=f"generate:{args.split}")
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            max_length = int(run_config.get("training", {}).get("max_length", 512))
            encoded = tokenizer(
                [_prompt(row) for row in batch_rows],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            for row, tokens in zip(batch_rows, new_tokens):
                raw_prediction = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                prediction = parse_prediction(raw_prediction)
                result = {
                    "id": row.get("id"),
                    "raw_prediction": raw_prediction,
                    "prediction": prediction,
                    "reference": row.get("answer", ""),
                    "is_impossible": row.get("is_impossible", False),
                    "has_gold_label": row.get("has_gold_label", True),
                    "split": args.split if args.data_file is None else str(data_file),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.update(len(batch_rows))
        progress.close()
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()

