$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage85-direct-log-stage84"
$Stage84 = "runs/stage84-fused-copy-stage83/fused-copy-calibrated-collapsed-hybrid.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage85" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_fused_residual_norm -v
if ($LASTEXITCODE -ne 0) { throw "Residual-normalization tests failed" }
& $Python scripts/prepare_stage85_residual_norm_calibrated_hybrid.py `
    --stage84 $Stage84 --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage85 qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage85 complete: inspect direct-log quality and resource gates; no test scoring."
