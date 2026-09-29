$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage43-exact-local-cache"
$Checkpoint = "runs/stage26-multi-token-s17/average-inference.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage43" }
New-Item -ItemType Directory -Path $Run | Out-Null
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_exact_local_cache -v
if ($LASTEXITCODE -ne 0) { throw "Exact-cache tests failed" }
& $Python scripts/scan_stage43_exact_local_cache.py --checkpoint $Checkpoint `
    --output "$Run/scan.json"
if ($LASTEXITCODE -ne 0) { throw "Stage43 scan failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage43 complete: exact local-cache validation diagnostic; no test scoring."
