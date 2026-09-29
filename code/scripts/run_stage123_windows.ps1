$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage123-gate-aware-calibration.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage123" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage123_gate_calibration.py `
    --neural "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --counts "runs/stage25-kneser-ney/counts-min2/checkpoint.pt" `
    --train-gate "runs/stage100-train-gate.json" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage123 screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage123 complete: inspect validation screen before any export."
