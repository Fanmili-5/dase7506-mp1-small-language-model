$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage51-width288-copy32-preflight"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage51" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_output_reallocation -v
if ($LASTEXITCODE -ne 0) { throw "Output-reallocation tests failed" }
& $Python scripts/preflight_stage33_depth.py --counts $Counts --baseline $Baseline --run-dir $Run `
    --config configs/stage51_width288_copy32.json `
    --architecture "8x288 Transformer, 8 heads, SwiGLU528, prefix-copy32, collapsed MKN .125"
if ($LASTEXITCODE -ne 0) { throw "Stage51 preflight failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage51 complete: output-reallocated width preflight; no test scoring."
