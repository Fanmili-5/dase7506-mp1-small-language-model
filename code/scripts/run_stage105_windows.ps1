$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage105-gated-singlepass"
$Stage103 = "runs/stage103-gated-order6-export/stage103-gated-order6.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage105" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/qualify_stage105_gated_singlepass.py `
    --stage103 $Stage103 --baseline $Baseline --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage105 qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage105 complete: inspect single-pass resource audit; no test scoring."
