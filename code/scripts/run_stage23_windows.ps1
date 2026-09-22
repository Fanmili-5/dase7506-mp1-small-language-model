$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage23-dynamic-gate"
if (Test-Path $Run) { throw "Refusing to overwrite Stage23" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_dynamic_gate -v
if ($LASTEXITCODE -ne 0) { throw "Dynamic-gate tests failed" }
& $Python scripts/fit_stage23_dynamic_gate.py `
    --neural runs/stage22-deep-supervision-s17/average-inference.pt `
    --counts runs/stage16-ngram-min3/checkpoint.pt --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Gate fitting failed" }
& $Python evaluate.py --checkpoint "$Run/dynamic-gate.pt" --device cpu --precision fp32 `
    --threads 4 --split validation --output "$Run/validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Dynamic-gate validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage23 complete: train-only gate plus validation score; no test scoring."
