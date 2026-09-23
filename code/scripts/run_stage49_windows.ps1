$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage49-hybrid-conv-preflight"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage49" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv -v
if ($LASTEXITCODE -ne 0) { throw "Hybrid-conv tests failed" }
& $Python scripts/preflight_stage49_hybrid_conv.py --counts $Counts --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage49 preflight failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage49 complete: hybrid attention/conv resource preflight; no test scoring."
