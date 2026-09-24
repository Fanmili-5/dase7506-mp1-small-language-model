$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$TrainGate = "runs/stage100-train-gate.json"
$Stage114 = "runs/stage114-order5-gate.json"
$Output = "runs/stage117-no-margin-gate.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage117" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/fit_stage117_no_margin_gate.py `
    --neural $Neural --counts $Counts --train-gate $TrainGate `
    --stage114 $Stage114 --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage117 train-only gate refit failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage117 complete: inspect training-fitted no-margin gate; no test scoring."
