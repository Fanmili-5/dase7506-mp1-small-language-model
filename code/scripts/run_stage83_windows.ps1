$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage83-norm-folded-calibrated-stage71"
$Stage80 = "runs/stage80-calibrated-stage71/calibrated-collapsed-hybrid.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage83" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_calibrated_hybrid_conv -v
if ($LASTEXITCODE -ne 0) { throw "Calibrated hybrid tests failed" }
& $Python scripts/prepare_stage83_norm_folded_calibrated_hybrid.py `
    --stage80 $Stage80 --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage83 qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage83 complete: inspect equivalent quality and resource gates; no test scoring."
