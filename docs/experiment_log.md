# Experiment log

## 2026-10-26 — Gate 1 audit

- Audit raw dataset hoàn tất.
- Grouped holdout seed 13 đã có manifest và hash.
- Truncation audit ở `max_length=512` hoàn tất cho Sailor2 và Qwen.
- Parser prediction đã được cố định trong `code/prediction_utils.py`.
- Calibration Sailor2 seed 13, 100 steps, tối đa 2.048 train samples và 256 dev samples đã hoàn tất.
- Đây là calibration, chưa phải kết quả khoa học và chưa được dùng để claim chất lượng.

## 2026-10-26 — Calibration hoàn tất

- Sailor2 seed 13, 100 steps: hoàn tất không OOM/NaN.
- Train loss cuối: `1.798`; eval loss trên 256 mẫu: `1.985`; train runtime khoảng `671.9s`.
- Sinh 32 prediction trên grouped dev thành công với parser mới.
- Calibration EM `0.34375`, F1 `0.41422`; chỉ là kiểm tra pipeline, không phải kết quả chính.
- Gate 1: `PASS` cho khả năng tái lập pipeline; Gate 2 chưa đánh giá vì chưa có đủ multi-seed baseline.

## 2026-10-26 — Main baseline started

- Sailor2-1B-Chat, seed 13, grouped train/dev, full one-epoch configuration.
- Evaluation and checkpoint frequency overridden to every 1,000 steps to control wall-clock overhead; this does not change the training data, seed, model, or optimization hyperparameters.
- Output: `checkpoints/baseline_sailor2_seed13`.
- Run đang thực hiện; chưa có metric chính thức.

### Last status check

- Progress observed: `450 / 2,837` optimizer steps (~15.9%).
- GPU: approximately 5.7 / 8.0 GiB used; process alive; no OOM/NaN reported.
- No checkpoint has been written yet because the first checkpoint interval is step 1,000.

## 2026-10-26 — Baseline restarted persistently

- The first foreground process ended before its first checkpoint, so its 450 steps were not resumable.
- Baseline Sailor2 seed 13 was restarted as a hidden background process with checkpoint interval 250 steps and per-run logs.
- Process ID at restart: `23320`.
- Standard output: `logs/baseline_sailor2_seed13.stdout.log`.
- Standard error: `logs/baseline_sailor2_seed13.stderr.log`.

- Baseline supervisor: `scripts/run_baseline_sequence.ps1`, process ID `23476`; waits for the current run and then trains Sailor2 seeds 37/73 and Qwen control seeds 13/37/73 sequentially.
- Evaluation supervisor: `scripts/run_evaluation_sequence.ps1`, process ID `936`; waits for all six baseline `run_metrics.json` files, then generates grouped-dev predictions and metrics.
- Telemetry supervisor: `scripts/run_telemetry_pilot.ps1`, process ID `16108`; waits for all baseline dev metrics, then runs six 200-step telemetry pilots with fixed seeds and JSONL gradient logs.

## 2026-10-26 — Throughput pilot

- The previous Sailor2 seed 13 process stopped after step 418; `checkpoint-250` remains valid and resumable.
- Added explicit speed-pilot overrides to `code/train_qlora_baseline.py` and resume support.
- Batch 2 / accumulation 4: `2.90s/step`, peak reserved VRAM `4.40 GiB`.
- Batch 4 / accumulation 2: `2.44s/step`, peak reserved VRAM `6.85 GiB`, no OOM.
- Selected batch 4 / accumulation 2 with gradient checkpointing disabled and TF32 enabled. Effective batch remains 8, so the baseline comparison stays matched while throughput improves.
- `scripts/run_baseline_sequence.ps1` now applies the same settings to every baseline and resumes the newest checkpoint when present.

## 2026-10-26 — Accelerated main run resumed

- Resumed Sailor2 seed 13 from `checkpoints/baseline_sailor2_seed13/checkpoint-250` with the selected throughput configuration.
- Current process: `26296`; baseline supervisor: `8472`; evaluation supervisor: `3484`; telemetry supervisor: `11892`.
- At the first live check, GPU utilization was `98%` and VRAM was approximately `7.9 / 8.0 GiB`; no OOM observed.

## 2026-10-26 — Sailor2 seed 13 completed

- One-epoch training and final grouped-dev evaluation completed successfully.
- Train loss: `1.2607`; final dev loss: `2.332`.
- Train wall-clock: `6,976s` (~1h56m) with the accelerated batch/TF32 configuration.
- Final adapter and `run_metrics.json` are in `checkpoints/baseline_sailor2_seed13`.
- The original supervisor treated a harmless Windows CUDA warning on stderr as a failure before seed 37. The supervisor was hardened to use isolated stdout/stderr files and process exit codes; seed 37 was then restarted successfully.

## 2026-10-27 — Multi-seed baseline complete (Gate 2 evidence)

All six baseline runs finished one full epoch (2,837 optimizer steps) on the grouped holdout, and predictions/metrics were generated on the same 2,866-row grouped dev split.

| Model | Seed | EM | F1 | Val loss | Train loss | Runtime (s) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sailor2-1B-Chat | 13 | 0.3343 | 0.4257 | 2.332 | 1.2607 | 6,976 | 5.58 |
| Sailor2-1B-Chat | 37 | 0.3416 | 0.4388 | 2.342 | 1.3092 | 6,549 | 5.58 |
| Sailor2-1B-Chat | 73 | 0.3402 | 0.4404 | 2.348 | 1.3096 | 8,608 | 5.58 |
| Qwen2.5-0.5B-Instruct | 13 | 0.3193 | 0.3722 | 2.427 | 1.9950 | 3,463 | 4.67 |
| Qwen2.5-0.5B-Instruct | 37 | 0.3217 | 0.3701 | 2.428 | 1.9946 | 3,628 | 4.67 |
| Qwen2.5-0.5B-Instruct | 73 | 0.3196 | 0.3700 | 2.429 | 1.9953 | 3,652 | 4.67 |

- Group means: Sailor2 EM `0.3387 ± 0.0039`, F1 `0.4350 ± 0.0081`; Qwen EM `0.3202 ± 0.0013`, F1 `0.3708 ± 0.0013`.
- Between-seed spread is small for both models, so the per-layer telemetry has to explain within-model risk rather than a large end-task variance.
- Sailor2 seed 13 was resumed from `checkpoint-250`; its runtime and train loss cover steps 250–2,837 while EM/F1/val loss cover the full grouped dev.
- Artifacts: `results/baseline_summary.json`, `results/*_dev_metrics.json`, `paper/baseline_results.md`.

## 2026-10-31 — Telemetry pilot v0 invalid, collector fixed

- The six 200-step telemetry pilots ran, but every JSONL record was empty: `"layers": {}` and `"total_gradient_norm": 0.0`.
- Root cause: `GradientTelemetryCallback` sampled in `on_step_end`, while `transformers` calls `model.zero_grad()` (trainer.py:1926) *before* firing that event (trainer.py:1929). Every `parameter.grad` was already `None`.
- Second defect found while fixing: sampling after `optimizer.step()` yields gradients that have already been clipped to `max_grad_norm=1.0`, so the total norm was pinned at exactly `1.0` and lost its magnitude.
- Fix in `code/gradient_telemetry.py`:
  - sample in `on_optimizer_step`, where the accumulated gradients still exist and the update has just been applied;
  - register `register_post_accumulate_grad_hook` on every trainable LoRA parameter to record the raw pre-clip per-module norm;
  - seed the parameter reference at `on_train_begin` so the first record reports a real update delta;
  - remove hooks in `on_train_end`.
- Smoke check (Sailor2, 20 steps, telemetry every 5): records now carry 336 module entries per step with non-zero per-module gradient and update norms, and total gradient norms differ from the clip threshold (`1.0150, 1.0077, 0.9111, 0.9401`).
- The six invalid files were archived to `results/invalid_telemetry_v0/` and the stale `checkpoints/telemetry_*` adapters were deleted; the six pilots were re-run with the fixed collector.

## 2026-10-31 — Telemetry collection overhead removed

- The first re-run of the fixed collector took `16s/step` instead of `2.4s/step`: the per-parameter hook called `.item()`, which forces a CUDA synchronisation twice per micro-batch (336 LoRA tensors × 2 = 672 syncs per optimizer step).
- Fix in `code/gradient_telemetry.py`:
  - the hook now stores the 0-dim norm tensor and never touches the host;
  - `on_step_begin` computes whether the upcoming step will be recorded, so hooks do nothing on non-sampled steps and the snapshot is cleared before each sampled step;
  - the record loop stacks the three norm columns and performs a single host transfer instead of one `float()` per parameter.
- Measured with the same 20-step Sailor2 smoke config: `35.0s` train runtime (`1.75s/step`) versus `16s/step` for the synchronising version, still 336 modules per record and pre-clip totals (`1.0126, 0.9837, 0.9034, 0.9289`).
- Added `code/analyze_telemetry.py`, which reads the pilot JSONL files and emits a per-module risk table (JSON plus `docs/telemetry_analysis.md`). It reports run quality, per-module gradient level/volatility, cross-seed spread, update-to-weight ratio, module-kind and depth aggregates, and a composite percentile rank used only for ordering.

## 2026-10-31 — Telemetry pilots complete, Gate 2 reports a negative result

- All six pilots finished: 20 records per run, 336 modules per record for Sailor2 and 168 for Qwen, no empty records, no OOM. GPU back to idle afterwards.
- Module ordering by gradient norm is nearly identical across seeds: Spearman `0.981–0.986` (Sailor2) and `0.983–0.986` (Qwen). MLP `gate/up` receives 3–5× the attention `q/k` gradient on both models.
- Training was stable: loss fell monotonically in every run (Sailor2 `2.12 → 1.18`, Qwen `2.78 → 1.94`), zero loss spikes at robust z > 3.5. The only anomaly is a batch-level gradient jump at step 180 of Sailor2 seed 73 (`1.77 → 4.65 → 1.80`, 195/336 modules), which recovers on the next record.
- Gate 2 was run exactly as frozen in `docs/gate2_protocol.md`. On the primary label `late_update_ratio` the predictor reaches ρ `0.939` (Sailor2 seed 73) and `0.958` (Qwen seed 73), with transfer ρ `0.908` and `0.922`, all CI lower bounds > 0 and p ≤ 0.0005.
- The pass does not survive the controls: `F_shape` alone (depth + module kind) scores ρ `0.917` / `0.930` / `0.919` / `0.935`, and `F_full` improves on it by only `+0.028 / +0.021 / −0.011 / −0.013`, below the pre-registered 0.05 margin.
- Exploratory checks recorded as such: `late_update_ratio` has 99.5% (Sailor2) and 99.6% (Qwen) of its variance *between* modules, so it is essentially a geometric constant; and a seed-residual label (`late − mean of the other two seeds`) is unpredictable from warm-up telemetry, with ρ in `[−0.10, 0.13]` for every feature set and both models.
- `late_grad_cv` is more dynamic but does not transfer: p = 0.064 and 0.075 > 0.01.
- Decision: do not open QGA-R/QGA-S on this evidence. Report the work as a stability audit plus a negative result for gradient-telemetry risk prediction at a 200-step / 2,048-sample budget, as allowed by `docs/roadmap_qga_lora_vi.md` §7.
- Artifacts: `results/telemetry_analysis.json`, `docs/telemetry_analysis.md`, `results/gate2_predictor.json`, `docs/gate2_results.md`.

