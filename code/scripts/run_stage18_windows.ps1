$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
& "$CodeRoot\.venv\Scripts\python.exe" scripts/run_stage18_regularization.py `
    --run-dir runs/stage18-regularization-s17 `
    --baseline runs/stage3-long-baseline-s17/checkpoint-best.pt `
    --reference runs/stage15-mechanisms-s17/F-copy-depth8/average-last5.pt `
    --stage17-screen runs/stage17-distillation-s17/screen.json
if ($LASTEXITCODE -ne 0) { throw "Stage18 failed; inspect screen.json and job logs." }
Write-Output "Stage18 finished; validation only, no test call."
