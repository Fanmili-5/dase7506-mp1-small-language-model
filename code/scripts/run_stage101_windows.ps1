$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage101-anchored-gate.json"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Base = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Extended = "runs/stage73-order6-b/counts/checkpoint.pt"
$Gate = "runs/stage100-train-gate.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage101" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage101_anchored_gate.py `
    --neural $Neural --base $Base --extended $Extended --train-gate $Gate `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage101 anchored gate screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage101 complete: bounded gate setting screen; no test scoring."
