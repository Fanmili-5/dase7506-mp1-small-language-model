$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage84-fused-copy-stage83"
$Stage83 = "runs/stage83-norm-folded-calibrated-stage71/norm-folded-calibrated-collapsed-hybrid.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage84" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_fused_copy_collapsed -v
if ($LASTEXITCODE -ne 0) { throw "Fused-copy tests failed" }
& $Python scripts/prepare_stage84_fused_copy_calibrated_hybrid.py `
    --stage83 $Stage83 --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage84 qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage84 complete: inspect equivalent quality and resource gates; no test scoring."
