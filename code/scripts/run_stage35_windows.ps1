$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage35-depth9-preflight"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage35" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/preflight_stage33_depth.py --counts $Counts --baseline $Baseline --run-dir $Run `
    --config configs/stage35_depth9_width240.json `
    --architecture "9x240 standard Transformer, 8 heads, prefix-copy64, collapsed MKN .125"
if ($LASTEXITCODE -ne 0) { throw "Stage35 preflight failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage35 complete: 9x240 resource preflight; no test scoring."
