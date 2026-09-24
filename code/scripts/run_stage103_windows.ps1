$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage103-gated-order6-export"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage73-order6-b/counts/checkpoint.pt"
$Gate = "runs/stage100-train-gate.json"
$Selection = "runs/stage102-train-gate-ablation.json"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage103" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/export_stage103_gated_hybrid.py `
    --neural $Neural --counts $Counts --train-gate $Gate `
    --selection $Selection --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage103 export/qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage103 complete: inspect exact candidate and resource gates; no test scoring."
