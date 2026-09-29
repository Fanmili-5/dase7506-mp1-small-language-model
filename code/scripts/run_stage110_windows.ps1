$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage73-order6-b/counts/checkpoint.pt"
$TrainGate = "runs/stage100-train-gate.json"
$Stage102 = "runs/stage102-train-gate-ablation.json"
$Output = "runs/stage110-low-cost-gate.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage110" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage110_low_cost_gate.py `
    --neural $Neural --counts $Counts --train-gate $TrainGate `
    --stage102 $Stage102 --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage110 screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage110 completed: inspect cheap gate variants before export; no test scoring."
