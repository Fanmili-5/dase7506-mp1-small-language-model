$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Root = "runs/stage73-order6"
$Base = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Neural = "runs/stage67-output-bias-s17/average.pt"
if (Test-Path $Root) { throw "Refusing to overwrite Stage73" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_stage73_order6 tests.test_mixture_aware -v
if ($LASTEXITCODE -ne 0) { throw "Order-6 extension tests failed" }
& $Python scripts/build_stage73_order6.py --base $Base --output-dir "$Root/counts"
if ($LASTEXITCODE -ne 0) { throw "Order-6 count build failed" }
& $Python scripts/scan_stage73_order6.py --neural $Neural --base $Base `
    --extended "$Root/counts/checkpoint.pt" --run-dir "$Root/scan"
if ($LASTEXITCODE -ne 0) { throw "Order-6 quality screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage73 complete: inspect order-6 quality screen; no resource or test claim."
