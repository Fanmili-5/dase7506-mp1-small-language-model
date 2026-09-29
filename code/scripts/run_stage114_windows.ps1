$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$TrainGate = "runs/stage100-train-gate.json"
$Output = "runs/stage114-order5-gate.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage114" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage114_order5_gate.py `
    --neural $Neural --counts $Counts --train-gate $TrainGate --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage114 screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage114 completed: inspect order-five gate scores; no test scoring."
