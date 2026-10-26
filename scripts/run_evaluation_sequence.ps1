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
    $exitCode = Invoke-PythonLogged -Arguments (@('code\generate_predictions.py') + $common) -LogPrefix "logs\$($run.Name)_generation"
    if ($exitCode -ne 0) {
        Write-Host "Batch generation failed for $($run.Name); retrying with batch size 1"
        $common = @('--adapter', $run.Adapter, '--data-file', 'data\splits\grouped_holdout\dev.jsonl', '--split', 'validation', '--max-new-tokens', '64', '--batch-size', '1', '--output', $prediction)
        $exitCode = Invoke-PythonLogged -Arguments (@('code\generate_predictions.py') + $common) -LogPrefix "logs\$($run.Name)_generation_retry"
    }
    if ($exitCode -ne 0) {
        throw "Generation failed for $($run.Name)"
    }
    $exitCode = Invoke-PythonLogged -Arguments @('code\evaluate_vi.py', $prediction, '--output', $metrics) -LogPrefix "logs\$($run.Name)_evaluation"
    if ($exitCode -ne 0) {
        throw "Evaluation failed for $($run.Name)"
    }
}

$exitCode = Invoke-PythonLogged -Arguments @('code\make_results_table.py') -LogPrefix 'logs\baseline_summary'
if ($exitCode -ne 0) {
    throw 'Baseline summary assembly failed'
}

Write-Host 'All baseline evaluations completed.'
