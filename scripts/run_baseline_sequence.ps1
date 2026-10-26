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

# Throughput configuration for the 8 GB RTX 3070 Ti.  batch 4 x accumulation 2
# preserves the original effective batch size (8) while removing activation
# checkpointing overhead.  Use the same settings for every baseline so the
# comparison remains fair.
$speedArgs = @(
    '--per-device-train-batch-size', '4',
    '--gradient-accumulation-steps', '2',
    '--no-gradient-checkpointing',
    '--tf32'
)

foreach ($run in $runs) {
    $metrics = Join-Path $run.Output 'run_metrics.json'
    if (Test-Path $metrics) {
        Write-Host "Skipping completed run $($run.Name)"
        continue
    }

    $logDir = "logs\$($run.Name)"
    $stdoutLog = "$logDir\stdout.log"
    $stderrLog = "$logDir\stderr.log"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $checkpoint = Get-ChildItem -Path $run.Output -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^checkpoint-\d+$' } |
        Sort-Object { [int]($_.Name -replace '^checkpoint-', '') } -Descending |
        Select-Object -First 1
    $commandArgs = @(
        '--config', $run.Config,
        '--train-file', 'data\splits\grouped_holdout\train.jsonl',
        '--eval-file', 'data\splits\grouped_holdout\dev.jsonl',
        '--output-dir', $run.Output,
        '--seed', ([int]($run.Name -replace '.*seed', '')),
        '--eval-steps', '1000',
        '--save-steps', '250',
        '--logging-dir', $logDir
    ) + $speedArgs
    if ($null -ne $checkpoint) {
        Write-Host "Resuming $($run.Name) from $($checkpoint.FullName)"
        $commandArgs += @('--resume-from-checkpoint', $checkpoint.FullName)
    }
    Write-Host "Starting $($run.Name)"
    # Use Start-Process so harmless CUDA/Transformers warnings on stderr do
    # not become terminating PowerShell errors under ErrorActionPreference=Stop.
    $process = Start-Process -FilePath 'python' `
        -ArgumentList (@('code\train_qlora_baseline.py') + $commandArgs) `
        -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Training failed for $($run.Name) with exit code $($process.ExitCode)"
    }
}

Write-Host 'All baseline runs completed.'
