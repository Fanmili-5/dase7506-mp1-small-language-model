$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage126b-onnx-feature-pilot"
if (Test-Path $Run) { throw "Refusing to overwrite Stage126" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/probe_stage126_onnx_features.py `
    --neural "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage126 ONNX pilot failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
