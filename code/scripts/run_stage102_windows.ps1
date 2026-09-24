$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage102-train-gate-ablation.json"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage73-order6-b/counts/checkpoint.pt"
$Gate = "runs/stage100-train-gate.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage102" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage102_train_gate_ablation.py `
    --neural $Neural --counts $Counts --train-gate $Gate --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage102 gate ablation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage102 complete: bounded gate ablation; no test scoring."
