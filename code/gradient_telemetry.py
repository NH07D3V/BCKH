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

    Sampling runs in ``on_optimizer_step``. The training loop calls
    ``model.zero_grad()`` before ``on_step_end``, so gradient tensors no
    longer exist by the time that event fires; a callback reading them there
    records empty layers and a zero gradient norm. Right after
    ``optimizer.step()`` the accumulated gradients are still present and the
    update has just been applied, which is exactly the pair we want.
    """

    def __init__(self, output: Path, every_steps: int = 10) -> None:
        self.output = Path(output)
        self.every_steps = max(1, int(every_steps))
        self.previous: dict[str, torch.Tensor] = {}
        self.grad_snapshot: dict[str, float] = {}
        self.handles: list = []
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def _capture_grad(self, name: str, parameter: torch.nn.Parameter) -> None:
        # Runs at the end of every backward, before the trainer clips the
        # gradient, so the recorded norm is the raw accumulated value.
        if parameter.grad is None:
            return
        self.grad_snapshot[name] = float(
            torch.linalg.vector_norm(parameter.grad.detach().float()).item()
        )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Seed the reference point so the first record already reports the
        # parameter delta since initialisation instead of a zero update.
        if model is None:
            return control
        self.previous = {
            name: parameter.detach().float().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            register = getattr(parameter, "register_post_accumulate_grad_hook", None)
            if register is None:
                continue
            # The trainer applies max_grad_norm before optimizer.step(), so
            # reading gradients after the step only yields the clipped values.
            self.handles.append(
                register(lambda param, name=name: self._capture_grad(name, param))
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return control

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        step = int(state.global_step) + 1
        if step <= 0 or step % self.every_steps:
            return control
        layer_stats: dict[str, dict[str, float]] = {}
        total_grad_sq = 0.0
        total_update_sq = 0.0
        total_param_sq = 0.0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad_norm = self.grad_snapshot.get(name)
            if grad_norm is None:
                if parameter.grad is None:
                    continue
                grad_norm = float(torch.linalg.vector_norm(parameter.grad.detach().float()).item())
            current = parameter.detach().float().cpu()
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
        if not layer_stats:
            return control
        record = {
            "step": step,
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
