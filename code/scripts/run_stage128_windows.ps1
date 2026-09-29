$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage128-quadratic-cheap-gate.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage128" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage128_quadratic_gate.py `
    --neural "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --counts "runs/stage25-kneser-ney/counts-min2/checkpoint.pt" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage128 train-fit/validation screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage128 complete: inspect quality before any export; no test scoring."
