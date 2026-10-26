param()

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'

$runs = @(
    @{ Name = 'sailor2_seed13'; Adapter = 'checkpoints\baseline_sailor2_seed13' },
    @{ Name = 'sailor2_seed37'; Adapter = 'checkpoints\baseline_sailor2_seed37' },
    @{ Name = 'sailor2_seed73'; Adapter = 'checkpoints\baseline_sailor2_seed73' },
    @{ Name = 'qwen_seed13'; Adapter = 'checkpoints\baseline_qwen_seed13' },
    @{ Name = 'qwen_seed37'; Adapter = 'checkpoints\baseline_qwen_seed37' },
    @{ Name = 'qwen_seed73'; Adapter = 'checkpoints\baseline_qwen_seed73' }
)

while ($true) {
    $pending = @($runs | Where-Object { -not (Test-Path (Join-Path $_.Adapter 'run_metrics.json')) })
    if ($pending.Count -eq 0) { break }
    Write-Host "Waiting for $($pending.Count) baseline runs to finish"
    Start-Sleep -Seconds 60
}

foreach ($run in $runs) {
    $prediction = "results\$($run.Name)_dev.jsonl"
    $metrics = "results\$($run.Name)_dev_metrics.json"
    if (Test-Path $metrics) {
        continue
    }
    New-Item -ItemType Directory -Force -Path 'results' | Out-Null
    $common = @('--adapter', $run.Adapter, '--data-file', 'data\splits\grouped_holdout\dev.jsonl', '--split', 'validation', '--max-new-tokens', '64', '--batch-size', '2', '--output', $prediction)
    & python code\generate_predictions.py @common 2>&1 | Tee-Object -FilePath "logs\$($run.Name)_generation.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Batch generation failed for $($run.Name); retrying with batch size 1"
        $common = @('--adapter', $run.Adapter, '--data-file', 'data\splits\grouped_holdout\dev.jsonl', '--split', 'validation', '--max-new-tokens', '64', '--batch-size', '1', '--output', $prediction)
        & python code\generate_predictions.py @common 2>&1 | Tee-Object -FilePath "logs\$($run.Name)_generation_retry.log"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Generation failed for $($run.Name)"
    }
    & python code\evaluate_vi.py $prediction --output $metrics 2>&1 | Tee-Object -FilePath "logs\$($run.Name)_evaluation.log"
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $($run.Name)"
    }
}

& python code\make_results_table.py 2>&1 | Tee-Object -FilePath 'logs\baseline_summary.log'
if ($LASTEXITCODE -ne 0) {
    throw 'Baseline summary assembly failed'
}

Write-Host 'All baseline evaluations completed.'
