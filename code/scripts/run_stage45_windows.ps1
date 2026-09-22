$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage45-width288-preflight"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage45" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_compute_reallocation -v
if ($LASTEXITCODE -ne 0) { throw "Compute-reallocation tests failed" }
& $Python scripts/preflight_stage33_depth.py --counts $Counts --baseline $Baseline --run-dir $Run `
    --config configs/stage45_width288_mlp1833.json `
    --architecture "8x288 Transformer, 8 heads, SwiGLU ratio 1.8333333, prefix-copy64, collapsed MKN .125"
if ($LASTEXITCODE -ne 0) { throw "Stage45 preflight failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage45 complete: compute-reallocated width preflight; no test scoring."
