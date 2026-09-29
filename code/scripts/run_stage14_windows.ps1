$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
& $Python scripts/run_architecture_screen.py `
    --run-dir runs/stage14-architecture-s17 `
    --baseline runs/stage3-long-baseline-s17/checkpoint-best.pt
if ($LASTEXITCODE -ne 0) { throw "Architecture screen failed; inspect screen.json and job logs." }
Write-Output "Stage 14 completed. Validation only; no test call."
