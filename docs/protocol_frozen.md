# Frozen protocol v0.1

Freeze time: 2026-10-26, before opening any held-out grouped test metrics.

## Primary question

Test whether early quantization residual and stochastic gradient geometry predict QLoRA run stability and final QA quality beyond a loss-only baseline, with held-out seeds/model and matched budget.

## Fixed assets

- Primary: `sail/Sailor2-1B-Chat`, revision `51b48ecd7c0629e4c79dc927a0445e6b671d8692`.
- Control: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dataset: `taidng/UIT-ViQuAD2.0`, revision `406f09a45cc106a8f7b7fd0c25078883fe58cb1f`.
- Labeled grouped split: `data/splits/grouped_holdout/`, seed 13, grouped by `(title, context)`.
- Main train/dev: `22,691 / 2,866` rows; grouped held-out test: `2,897` rows.
- Original HF test remains unlabeled and is excluded from EM/F1 selection.

## Fixed training recipe

- QLoRA 4-bit NF4 with double quantization.
- LoRA rank 8, alpha 16, dropout 0.05.
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- Sequence length 512; **effective batch size 8**.
- Main throughput configuration: per-device batch 4, gradient accumulation 2, gradient checkpointing disabled, and TF32 enabled when supported. This keeps the effective batch and data/step budget fixed while reducing activation-recompute overhead; every primary/control run uses the same settings.
- Learning rate `2e-4`; one epoch; warmup ratio 0.03.
- Seeds: 13, 37, 73 for the primary baseline and control.
- Full baseline uses the same recipe for both models; only local model/revision and output directory differ.
- Evaluation frequency is an operational override of 1,000 steps and checkpoint frequency is 250 steps for long runs; these do not change optimization.

## Fixed formatting and scoring

- Prompt template is the `### Hướng dẫn / Đoạn văn / Câu hỏi / Trả lời` template in the training and generation scripts.
- `code/prediction_utils.py` is the sole parser. Save both raw and parsed prediction.
- Metrics: Exact Match and token-level F1 from `code/evaluate_vi.py`.
- Report answerable, unanswerable and overall metrics when the analysis script supports them; do not manually edit prediction files.
- Test metrics may be computed once only after predictor, policy, parser and hyperparameters are frozen.

## Required evidence before QGA-P

1. Baseline completed for Sailor2 and Qwen with three seeds.
2. Run-to-run variance and failure/loss-spike behavior are measured.
3. Telemetry features are defined before looking at held-out test scores.
4. Predictor is evaluated out-of-sample and compared with loss-only, residual-only and gradient-only controls.
5. Any rank policy uses the same total rank, tokens, steps, memory and measurement overhead as uniform QLoRA.

Gradient telemetry for the pilot is append-only JSONL from `code/gradient_telemetry.py`; it records LoRA gradient norm and update-to-weight ratio without changing optimizer behavior. Quantization residual measurement remains a separate, explicitly documented analysis step.

If these conditions fail, the paper is reported as a reproducible stability audit or negative result; no algorithmic novelty claim is made.

## Current execution

- Gate 1: PASS for asset, data, tokenization, parser and calibration pipeline.
- Main run in progress: `checkpoints/baseline_sailor2_seed13`.
- Held-out grouped test is still locked.
