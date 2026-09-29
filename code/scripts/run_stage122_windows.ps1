$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage122-folded-order5-gate"
if (Test-Path $Run) { throw "Refusing to overwrite Stage122" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/qualify_stage122_folded_gate.py `
    --stage115 "runs/stage115-order5-gate-export/stage115-order5-gate.pt" `
    --baseline "runs/stage3-long-baseline-s17/checkpoint-best.pt" `
    --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage122 export/qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage122 completed; inspect formal qualification and do not score test."
