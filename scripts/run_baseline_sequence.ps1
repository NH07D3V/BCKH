param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'

if ($WaitForPid -gt 0) {
    $existing = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Host "Waiting for existing training process $WaitForPid"
        Wait-Process -Id $WaitForPid
    }
}

$runs = @(
    @{ Name = 'sailor2_seed13'; Config = 'configs\sailor2_qlora_vi.yaml'; Output = 'checkpoints\baseline_sailor2_seed13' },
    @{ Name = 'sailor2_seed37'; Config = 'configs\sailor2_qlora_vi.yaml'; Output = 'checkpoints\baseline_sailor2_seed37' },
    @{ Name = 'sailor2_seed73'; Config = 'configs\sailor2_qlora_vi.yaml'; Output = 'checkpoints\baseline_sailor2_seed73' },
    @{ Name = 'qwen_seed13'; Config = 'configs\baseline_qlora_vi.yaml'; Output = 'checkpoints\baseline_qwen_seed13' },
    @{ Name = 'qwen_seed37'; Config = 'configs\baseline_qlora_vi.yaml'; Output = 'checkpoints\baseline_qwen_seed37' },
    @{ Name = 'qwen_seed73'; Config = 'configs\baseline_qlora_vi.yaml'; Output = 'checkpoints\baseline_qwen_seed73' }
)

foreach ($run in $runs) {
    $metrics = Join-Path $run.Output 'run_metrics.json'
    if (Test-Path $metrics) {
        Write-Host "Skipping completed run $($run.Name)"
        continue
    }

    $logDir = "logs\$($run.Name)"
    $stdoutLog = "$logDir\stdout.log"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Write-Host "Starting $($run.Name)"
    & python code\train_qlora_baseline.py `
        --config $run.Config `
        --train-file data\splits\grouped_holdout\train.jsonl `
        --eval-file data\splits\grouped_holdout\dev.jsonl `
        --output-dir $run.Output `
        --seed ([int]($run.Name -replace '.*seed', '')) `
        --eval-steps 1000 `
        --save-steps 250 `
        --logging-dir $logDir 2>&1 | Tee-Object -FilePath $stdoutLog
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $($run.Name) with exit code $LASTEXITCODE"
    }
}

Write-Host 'All baseline runs completed.'
