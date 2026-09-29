$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage28-weight-averages"
$Stage22 = "runs/stage22-deep-supervision-s17"
$H = "runs/stage18-regularization-s17/H-embedding-drop/average-inference.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage28" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_deep_supervision tests.test_checkpoint_average tests.test_weight_average -v
if ($LASTEXITCODE -ne 0) { throw "Averaging tests failed" }
& $Python scripts/screen_stage28_weight_averages.py --stage22-dir $Stage22 --stage18-h $H --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage28 averaging screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage28 complete: averaging/soup validation screen; no test scoring."
