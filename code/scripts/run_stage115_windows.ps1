$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage115-order5-gate-export"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$TrainGate = "runs/stage100-train-gate.json"
$Selection = "runs/stage114-order5-gate.json"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage115" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/qualify_stage115_order5_gate.py `
    --neural $Neural --counts $Counts --train-gate $TrainGate `
    --selection $Selection --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage115 export/preflight failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage115 one-repeat preflight done; not a final qualification."
