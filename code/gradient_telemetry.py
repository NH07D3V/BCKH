"""Low-overhead LoRA gradient telemetry for pilot stability analysis."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import TrainerCallback


def _layer_key(name: str) -> str:
    for marker in (".lora_A", ".lora_B"):
        if marker in name:
            return name.split(marker, 1)[0]
    return name.rsplit(".", 1)[0]


class GradientTelemetryCallback(TrainerCallback):
    """Record gradient norms and update-to-weight ratios at fixed intervals.

    This is a diagnostic callback, not an optimizer. It leaves training
    behavior unchanged and writes append-only JSONL records.
    """

    def __init__(self, output: Path, every_steps: int = 10) -> None:
        self.output = output
        self.every_steps = max(1, int(every_steps))
        self.previous: dict[str, torch.Tensor] = {}
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0 or state.global_step % self.every_steps:
            return control
        layer_stats: dict[str, dict[str, float]] = {}
        total_grad_sq = 0.0
        total_update_sq = 0.0
        total_param_sq = 0.0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            current = parameter.detach().float().cpu()
            grad_norm = float(torch.linalg.vector_norm(grad).item())
            param_norm = float(torch.linalg.vector_norm(current).item())
            previous = self.previous.get(name)
            update_norm = float(torch.linalg.vector_norm(current - previous).item()) if previous is not None else 0.0
            self.previous[name] = current
            key = _layer_key(name)
            bucket = layer_stats.setdefault(key, {"gradient_norm_sq": 0.0, "update_norm_sq": 0.0, "parameter_norm_sq": 0.0, "parameters": 0})
            bucket["gradient_norm_sq"] += grad_norm * grad_norm
            bucket["update_norm_sq"] += update_norm * update_norm
            bucket["parameter_norm_sq"] += param_norm * param_norm
            bucket["parameters"] += int(parameter.numel())
            total_grad_sq += grad_norm * grad_norm
            total_update_sq += update_norm * update_norm
            total_param_sq += param_norm * param_norm
        record = {
            "step": int(state.global_step),
            "total_gradient_norm": total_grad_sq ** 0.5,
            "total_update_norm": total_update_sq ** 0.5,
            "update_to_weight_ratio": total_update_sq ** 0.5 / (total_param_sq ** 0.5 + 1e-12),
            "layers": {
                key: {
                    "gradient_norm": value["gradient_norm_sq"] ** 0.5,
                    "update_norm": value["update_norm_sq"] ** 0.5,
                    "parameter_norm": value["parameter_norm_sq"] ** 0.5,
                    "update_to_weight_ratio": value["update_norm_sq"] ** 0.5 / (value["parameter_norm_sq"] ** 0.5 + 1e-12),
                    "parameters": value["parameters"],
                }
                for key, value in layer_stats.items()
            },
        }
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return control
