$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage135-successor-transfer"
if (Test-Path $Run) { throw "Refusing to overwrite Stage135" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_successor_only tests.test_stage135_successor_gate -q
if ($LASTEXITCODE -ne 0) { throw "Successor causality tests failed" }
& $Python scripts/screen_stage135_successor_transfer.py `
    --checkpoint "runs/stage115-order5-gate-export/stage115-order5-gate.pt" `
    --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage135 validation transfer screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage135 validation-only successor-copy transfer complete."
