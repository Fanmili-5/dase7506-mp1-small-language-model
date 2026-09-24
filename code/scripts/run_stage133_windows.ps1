$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage133-deployable-local-cache-pilot"
$Checkpoint = "runs/stage115-order5-gate-export/stage115-order5-gate.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage133" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_exact_local_cache tests.test_stage133_exact_local_cache -v
if ($LASTEXITCODE -ne 0) { throw "Local-cache tests failed" }
New-Item -ItemType Directory -Path $Run | Out-Null
& $Python scripts/probe_stage133_local_cache_cpu.py --checkpoint $Checkpoint `
    --output "$Run/probe.json"
if ($LASTEXITCODE -ne 0) { throw "Stage133 pilot failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage133 local-cache CPU pilot complete; no test scoring."
