$ErrorActionPreference = "Stop"
Set-Location "C:\Users\23223\mp1-challenge-20260919\repo\code"
$Python = ".venv\Scripts\python.exe"
$Output = "runs/stage127-int8-feature-diagnostic.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage127" }
& $Python scripts/probe_stage127_int8_features.py `
    --neural "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage127 diagnostic failed" }
Write-Output "Stage127 input-only diagnostic done; no qualification or test scoring."
