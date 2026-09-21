param(
    [string]$Config = "configs/student_modern.json"
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run scripts/setup_windows_cuda.ps1 first."
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunDir = "runs/smoke-$Stamp"
& $Python train_experiment.py `
    --implementation student `
    --config $Config `
    --device cuda `
    --precision auto `
    --steps 10 `
    --micro-batch-size 4 `
    --warmup-steps 2 `
    --log-every 1 `
    --save-every 5 `
    --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "CUDA smoke training failed: $RunDir" }

& $Python evaluate.py --checkpoint "$RunDir/checkpoint.pt" --device cpu --precision fp32 --split validation
if ($LASTEXITCODE -ne 0) { throw "CPU FP32 validation failed: $RunDir" }
Write-Host "Smoke test complete: $RunDir" -ForegroundColor Green
