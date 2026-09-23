$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage58-local-heavy-preflight"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage58" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv tests.test_hybrid_conv_rdrop -v
if ($LASTEXITCODE -ne 0) { throw "Hybrid-conv tests failed" }
& $Python scripts/preflight_stage58_local_heavy.py `
    --counts $Counts --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage58 preflight failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage58 complete: resource preflight only; no test scoring."
