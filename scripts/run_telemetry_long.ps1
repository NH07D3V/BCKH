param()

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = 'utf-8'

function Invoke-PythonLogged {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPrefix
    )
    $process = Start-Process -FilePath 'python' `
        -ArgumentList $Arguments `
        -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput "$LogPrefix.stdout.log" `
        -RedirectStandardError "$LogPrefix.stderr.log" `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    return [int]$process.ExitCode
}

$runs = @(
    @{ Name = 'sailor2_seed13'; Config = 'configs\sailor2_qlora_vi.yaml' },
    @{ Name = 'sailor2_seed37'; Config = 'configs\sailor2_qlora_vi.yaml' },
    @{ Name = 'sailor2_seed73'; Config = 'configs\sailor2_qlora_vi.yaml' },
    @{ Name = 'qwen_seed13'; Config = 'configs\baseline_qlora_vi.yaml' },
    @{ Name = 'qwen_seed37'; Config = 'configs\baseline_qlora_vi.yaml' },
    @{ Name = 'qwen_seed73'; Config = 'configs\baseline_qlora_vi.yaml' }
)

foreach ($run in $runs) {
    $telemetry = "results\telemetry_long_$($run.Name).jsonl"
    if (Test-Path $telemetry) {
        continue
    }
    $output = "checkpoints\telemetry_long_$($run.Name)"
    $logDir = "logs\telemetry_long_$($run.Name)"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $seed = [int]($run.Name -replace '.*seed', '')
    $arguments = @(
        'code\train_qlora_baseline.py',
        '--config', $run.Config,
        '--max-steps', '1000',
        '--max-eval-samples', '256',
        '--train-file', 'data\splits\grouped_holdout\train.jsonl',
        '--eval-file', 'data\splits\grouped_holdout\dev.jsonl',
        '--output-dir', $output,
        '--seed', $seed,
        '--eval-steps', '250',
        '--save-steps', '1000',
        '--logging-dir', $logDir,
        '--telemetry-output', $telemetry,
        '--telemetry-every', '10',
        '--per-device-train-batch-size', '4',
        '--gradient-accumulation-steps', '2',
        '--no-gradient-checkpointing',
        '--tf32'
    )
    $exitCode = Invoke-PythonLogged -Arguments $arguments -LogPrefix "$logDir\run"
    if ($exitCode -ne 0) {
        throw "Long telemetry pilot failed for $($run.Name)"
    }
}

Write-Host 'All long telemetry pilots completed.'
