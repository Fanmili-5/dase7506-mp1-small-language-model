$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage100-train-gate.json"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Base = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Extended = "runs/stage73-order6-b/counts/checkpoint.pt"
if (Test-Path $Output) { throw "Refusing to overwrite Stage100" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/fit_stage100_train_gate.py `
    --neural $Neural --base $Base --extended $Extended --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage100 train-only gate failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage100 complete: train-only gate fit and validation screen; no test scoring."
