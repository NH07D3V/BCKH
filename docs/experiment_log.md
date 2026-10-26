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
