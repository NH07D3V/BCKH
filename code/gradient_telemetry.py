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

    Norms are kept on the GPU as 0-dim tensors and only pulled to the host on
    the steps that are written. Calling ``.item()`` inside the per-parameter
    hook synchronises with the CUDA queue twice per micro-batch, which turned
    a 2.4 s step into a 16 s step on the first attempt.
    """

    def __init__(self, output: Path, every_steps: int = 10) -> None:
        self.output = Path(output)
        self.every_steps = max(1, int(every_steps))
        self.previous: dict[str, torch.Tensor] = {}
        self.grad_snapshot: dict[str, torch.Tensor] = {}
        self.handles: list = []
        self.capture_this_step = False
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def _capture_grad(self, name: str, parameter: torch.nn.Parameter) -> None:
        # Runs at the end of every backward, before the trainer clips the
        # gradient, so the recorded norm is the raw accumulated value.
        if not self.capture_this_step or parameter.grad is None:
            return
        self.grad_snapshot[name] = torch.linalg.vector_norm(
            parameter.grad.detach().float()
        )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Seed the reference point so the first record already reports the
        # parameter delta since initialisation instead of a zero update.
        if model is None:
            return control
        self.previous = {
            name: parameter.detach().float().clone()
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

    def on_step_begin(self, args, state, control, **kwargs):
        # ``on_step_begin`` fires once per optimizer step, before the first
        # micro-batch backward. ``global_step`` still counts completed steps.
        upcoming = int(state.global_step) + 1
        self.capture_this_step = upcoming % self.every_steps == 0
        if self.capture_this_step:
            self.grad_snapshot.clear()
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
        keys: list[str] = []
        grad_norms: list[torch.Tensor] = []
        update_norms: list[torch.Tensor] = []
        param_norms: list[torch.Tensor] = []
        counts: list[int] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad_norm = self.grad_snapshot.get(name)
            if grad_norm is None:
                if parameter.grad is None:
                    continue
                grad_norm = torch.linalg.vector_norm(parameter.grad.detach().float())
            current = parameter.detach().float()
            previous = self.previous.get(name)
            if previous is None:
                update_norm = torch.zeros((), device=current.device, dtype=current.dtype)
            else:
                update_norm = torch.linalg.vector_norm(current - previous)
            self.previous[name] = current.clone()
            keys.append(_layer_key(name))
            grad_norms.append(grad_norm)
            update_norms.append(update_norm)
            param_norms.append(torch.linalg.vector_norm(current))
            counts.append(int(parameter.numel()))
        if not keys:
            return control
        # One host transfer per column instead of one CUDA synchronisation per
        # parameter: 336 .item() calls per record still serialise the queue.
        columned = torch.stack(
            [torch.stack(grad_norms), torch.stack(update_norms), torch.stack(param_norms)]
        ).cpu().tolist()
        gradient_values, update_values, parameter_values = columned
        layer_stats: dict[str, dict[str, float]] = {}
        total_grad_sq = 0.0
        total_update_sq = 0.0
        total_param_sq = 0.0
        for key, grad_norm, update_norm, param_norm, count in zip(
            keys, gradient_values, update_values, parameter_values, counts
        ):
            bucket = layer_stats.setdefault(
                key,
                {
                    "gradient_norm_sq": 0.0,
                    "update_norm_sq": 0.0,
                    "parameter_norm_sq": 0.0,
                    "parameters": 0,
                },
            )
            bucket["gradient_norm_sq"] += grad_norm * grad_norm
            bucket["update_norm_sq"] += update_norm * update_norm
            bucket["parameter_norm_sq"] += param_norm * param_norm
            bucket["parameters"] += count
            total_grad_sq += grad_norm * grad_norm
            total_update_sq += update_norm * update_norm
            total_param_sq += param_norm * param_norm
        record = {
            "step": step,
            "total_gradient_norm": total_grad_sq ** 0.5,
            "total_update_norm": total_update_sq ** 0.5,
            "update_to_weight_ratio": total_update_sq ** 0.5
            / (total_param_sq ** 0.5 + 1e-12),
            "layers": {
                key: {
                    "gradient_norm": value["gradient_norm_sq"] ** 0.5,
                    "update_norm": value["update_norm_sq"] ** 0.5,
                    "parameter_norm": value["parameter_norm_sq"] ** 0.5,
                    "update_to_weight_ratio": value["update_norm_sq"] ** 0.5
                    / (value["parameter_norm_sq"] ** 0.5 + 1e-12),
                    "parameters": value["parameters"],
                }
                for key, value in layer_stats.items()
            },
        }
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return control
