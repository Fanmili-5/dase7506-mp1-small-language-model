$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage70-balanced-hybrid-preflight"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage70" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv tests.test_hybrid_conv_rdrop -v
if ($LASTEXITCODE -ne 0) { throw "Balanced hybrid tests failed" }
& $Python scripts/preflight_stage58_local_heavy.py `
    --counts $Counts --baseline $Baseline --run-dir $Run `
    --config configs/stage70_balanced_hybrid_rdrop.json `
    --architecture "8x304, attention layers 1/4/7, five gated causal-conv layers, SwiGLU684, prefix-copy64, collapsed MKN .075"
if ($LASTEXITCODE -ne 0) { throw "Stage70 preflight failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage70 complete: balanced-hybrid resource preflight only; no test scoring."
