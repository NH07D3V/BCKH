$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'
python code\verify_assets.py
python code\audit_viquad.py
python code\train_qlora_baseline.py --config configs\sailor2_qlora_vi.yaml --max-steps 2 --max-train-samples 8 --max-eval-samples 4 --output-dir checkpoints\smoke_sailor2_qlora_vi --seed 13
python code\generate_predictions.py --adapter checkpoints\smoke_sailor2_qlora_vi --split validation --limit 4 --max-new-tokens 16 --output results\smoke_sailor2_predictions.jsonl
python code\evaluate_vi.py results\smoke_sailor2_predictions.jsonl
python code\train_qlora_baseline.py --config configs\baseline_qlora_vi.yaml --max-steps 2 --max-train-samples 8 --max-eval-samples 4 --output-dir checkpoints\smoke_qwen_qlora_vi --seed 13
python code\generate_predictions.py --adapter checkpoints\smoke_qwen_qlora_vi --split validation --limit 4 --max-new-tokens 16 --output results\smoke_qwen_predictions.jsonl
python code\evaluate_vi.py results\smoke_qwen_predictions.jsonl

