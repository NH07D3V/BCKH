"""Minimal, reproducible QLoRA baseline for Vietnamese extractive QA."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _make_text(example: dict, eos: str) -> str:
    answer = "[KHÔNG CÓ ĐÁP ÁN]" if example.get("is_impossible") else (example.get("answer") or "")
    return (
        "### Hướng dẫn\nTrả lời câu hỏi bằng thông tin trong đoạn văn.\n"
        f"### Đoạn văn\n{example.get('context', '')}\n"
        f"### Câu hỏi\n{example.get('question', '')}\n"
        f"### Trả lời\n{answer}{eos}"
    )


def _training_args(kwargs: dict, has_eval: bool) -> TrainingArguments:
    """Build arguments across the Transformers 4.x/5.x naming changes."""
    supported = inspect.signature(TrainingArguments.__init__).parameters
    values = {key: value for key, value in kwargs.items() if key in supported}
    strategy_key = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    values[strategy_key] = "steps" if has_eval else "no"
    return TrainingArguments(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_qlora_vi.yaml")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--eval-steps", type=int, default=None, help="Optional override for evaluation frequency.")
    parser.add_argument("--save-steps", type=int, default=None, help="Optional override for checkpoint frequency.")
    parser.add_argument("--telemetry-output", type=Path, default=None, help="Optional JSONL path for LoRA gradient telemetry.")
    parser.add_argument("--telemetry-every", type=int, default=10, help="Telemetry interval in optimizer steps.")
    parser.add_argument("--logging-dir", type=Path, default=None, help="Optional per-run logging directory.")
    parser.add_argument("--no-4bit", action="store_true", help="Debug fallback; not the QLoRA research condition.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None, help="Optional throughput override for a speed pilot.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None, help="Optional evaluation batch-size override.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Optional effective-batch override.")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable activation checkpointing for a speed/memory pilot.")
    parser.add_argument("--dataloader-num-workers", type=int, default=None, help="Optional DataLoader worker count.")
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 matrix multiplication on supported NVIDIA GPUs.")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None, help="Resume Trainer/optimizer state from a checkpoint directory.")
    args = parser.parse_args()
    cfg = _load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    model_path = ROOT / model_cfg["local_dir"]
    data_path = ROOT / data_cfg["processed_dir"]
    train_file = args.train_file or (ROOT / data_cfg.get("train_file", data_path / f"{data_cfg['train_split']}.jsonl"))
    eval_file = args.eval_file or (ROOT / data_cfg.get("validation_file", data_path / f"{data_cfg['validation_split']}.jsonl"))
    if not train_file.is_absolute():
        train_file = ROOT / train_file
    if not eval_file.is_absolute():
        eval_file = ROOT / eval_file
    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}; run prepare_viquad.py first.")

    data_files = {"train": str(train_file)}
    if eval_file.exists():
        data_files["validation"] = str(eval_file)
    raw = load_dataset("json", data_files=data_files)
    for split_name in list(raw):
        if "has_gold_label" in raw[split_name].column_names:
            unlabeled = [index for index, flag in enumerate(raw[split_name]["has_gold_label"]) if not flag]
            if unlabeled:
                raise ValueError(f"{split_name} contains {len(unlabeled)} unlabeled rows; do not train/evaluate on them.")
    if args.max_train_samples:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples, len(raw["train"]))))
    if "validation" in raw and args.max_eval_samples:
        raw["validation"] = raw["validation"].select(range(min(args.max_eval_samples, len(raw["validation"]))))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        texts = [_make_text(row, tokenizer.eos_token or "") for row in _rows(batch)]
        return tokenizer(texts, truncation=True, max_length=int(train_cfg["max_length"]))

    def _rows(batch: dict) -> list[dict]:
        keys = list(batch)
        return [{key: batch[key][i] for key in keys} for i in range(len(batch[keys[0]]))]

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    use_4bit = bool(cfg["quantization"]["load_in_4bit"]) and not args.no_4bit
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("QLoRA 4-bit requires a CUDA runtime. Use a CUDA/WSL2 environment or --no-4bit for a debug-only run.")

    quantization_config = None
    if use_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=bool(cfg["quantization"]["bnb_4bit_use_double_quant"]),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    use_gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True)) and not args.no_gradient_checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    output_dir = args.output_dir or (ROOT / train_cfg["output_dir"])
    log_dir = args.logging_dir or (ROOT / train_cfg["logging_dir"])
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps if args.max_steps > 0 else -1
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(args.per_device_train_batch_size or train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size or train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps or train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(args.save_steps if args.save_steps is not None else train_cfg["save_steps"]),
        eval_steps=int(args.eval_steps if args.eval_steps is not None else train_cfg["eval_steps"]),
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        max_steps=max_steps,
        seed=int(args.seed if args.seed is not None else cfg["seed_list"][0]),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=int(args.dataloader_num_workers) if args.dataloader_num_workers is not None else 0,
        tf32=bool(args.tf32),
    )
    training_args = _training_args(kwargs, has_eval="validation" in tokenized)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    if args.telemetry_output is not None:
        from gradient_telemetry import GradientTelemetryCallback

        telemetry_path = args.telemetry_output if args.telemetry_output.is_absolute() else ROOT / args.telemetry_output
        trainer.add_callback(GradientTelemetryCallback(telemetry_path, every_steps=args.telemetry_every))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is not None and not resume_checkpoint.is_absolute():
        resume_checkpoint = ROOT / resume_checkpoint
    train_output = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    elapsed = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    run_metrics = {
        "train_metrics": train_output.metrics,
        "wall_clock_seconds": elapsed,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        "seed": int(args.seed if args.seed is not None else cfg["seed_list"][0]),
        "max_steps_override": args.max_steps if args.max_steps > 0 else None,
        "train_file": str(train_file),
        "eval_file": str(eval_file) if eval_file.exists() else None,
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(run_metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

