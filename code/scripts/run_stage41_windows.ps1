$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage41-continuous-cache"
$Checkpoint = "runs/stage26-multi-token-s17/average-inference.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage41" }
New-Item -ItemType Directory -Path $Run | Out-Null
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_continuous_cache -v
if ($LASTEXITCODE -ne 0) { throw "Continuous-cache tests failed" }
& $Python scripts/scan_stage41_continuous_cache.py --checkpoint $Checkpoint `
    --output "$Run/scan.json"
if ($LASTEXITCODE -ne 0) { throw "Stage41 scan failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage41 complete: validation-only continuous-cache diagnostic; no test scoring."
