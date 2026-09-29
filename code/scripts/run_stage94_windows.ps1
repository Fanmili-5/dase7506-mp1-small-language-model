$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage94-distilled-calibration"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage94" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage94_distilled_calibration.py `
    --neural $Neural --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage94 recalibration failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage94 complete: inspect bounded recalibration; no test scoring."
