$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
& "$CodeRoot\.venv\Scripts\python.exe" scripts/run_stage15_screen.py `
    --run-dir runs/stage15-mechanisms-s17 `
    --baseline runs/stage3-long-baseline-s17/checkpoint-best.pt `
    --reference runs/stage14-architecture-s17/B-copy/average-last5.pt
if ($LASTEXITCODE -ne 0) { throw "Stage 15 failed; inspect screen.json and job logs." }
Write-Output "Stage 15 completed. Validation only; no test call."
