param()

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'

$runs = @(
    @{ Name = 'sailor2_seed13'; Config = 'configs\sailor2_qlora_vi.yaml' },
    @{ Name = 'sailor2_seed37'; Config = 'configs\sailor2_qlora_vi.yaml' },
    @{ Name = 'sailor2_seed73'; Config = 'configs\sailor2_qlora_vi.yaml' },
    @{ Name = 'qwen_seed13'; Config = 'configs\baseline_qlora_vi.yaml' },
    @{ Name = 'qwen_seed37'; Config = 'configs\baseline_qlora_vi.yaml' },
    @{ Name = 'qwen_seed73'; Config = 'configs\baseline_qlora_vi.yaml' }
)

while ($true) {
    $pending = @($runs | Where-Object { -not (Test-Path "results\$($_.Name)_dev_metrics.json") })
    if ($pending.Count -eq 0) { break }
    Write-Host "Waiting for $($pending.Count) baseline evaluations before telemetry pilot"
    Start-Sleep -Seconds 60
}

foreach ($run in $runs) {
    $telemetry = "results\telemetry_$($run.Name).jsonl"
    if (Test-Path $telemetry) {
        continue
    }
    $output = "checkpoints\telemetry_$($run.Name)"
    $logDir = "logs\telemetry_$($run.Name)"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $seed = [int]($run.Name -replace '.*seed', '')
    & python code\train_qlora_baseline.py `
        --config $run.Config `
        --max-steps 200 `
        --max-train-samples 2048 `
        --max-eval-samples 256 `
        --train-file data\splits\grouped_holdout\train.jsonl `
        --eval-file data\splits\grouped_holdout\dev.jsonl `
        --output-dir $output `
        --seed $seed `
        --eval-steps 200 `
        --save-steps 200 `
        --logging-dir $logDir `
        --telemetry-output $telemetry `
        --telemetry-every 10 2>&1 | Tee-Object -FilePath "$logDir\stdout.log"
    if ($LASTEXITCODE -ne 0) {
        throw "Telemetry pilot failed for $($run.Name)"
    }
}

Write-Host 'All telemetry pilots completed.'
