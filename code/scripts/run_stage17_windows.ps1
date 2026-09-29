$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
& "$CodeRoot\.venv\Scripts\python.exe" scripts/audit_stage15_screen.py `
    --run-dir runs/stage15-mechanisms-s17 `
    --output results/stage15-audit.json
if ($LASTEXITCODE -ne 0) { throw "Stage15 evidence audit failed; no Stage17 training started." }
& "$CodeRoot\.venv\Scripts\python.exe" scripts/run_stage17_distillation.py `
    --run-dir runs/stage17-distillation-s17 `
    --baseline runs/stage3-long-baseline-s17/checkpoint-best.pt `
    --stage15-screen runs/stage15-mechanisms-s17/screen.json --execute
if ($LASTEXITCODE -ne 0) { throw "Stage17 failed; inspect screen.json and job logs." }
Write-Output "Stage17 finished its bounded plan. Validation only; no test call."
