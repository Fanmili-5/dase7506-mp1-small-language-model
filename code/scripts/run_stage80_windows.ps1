$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage80-calibrated-stage71"
$Neural = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$Screen = "runs/stage79-stage71-calibration-refinement/screen.json"
if (Test-Path $Run) { throw "Refusing to overwrite Stage80" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_calibrated_hybrid_conv -v
if ($LASTEXITCODE -ne 0) { throw "Calibrated hybrid tests failed" }
& $Python scripts/prepare_stage80_calibrated_hybrid.py `
    --neural $Neural --counts $Counts --baseline $Baseline `
    --screen $Screen --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage80 qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage80 complete: inspect calibrated quality and resource gates; no test scoring."
