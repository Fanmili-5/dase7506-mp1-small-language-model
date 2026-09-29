$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage60-deployable-gate"
$Neural = "runs/stage56-hybrid-conv-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage60" }
New-Item -ItemType Directory -Path $Run | Out-Null
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/analyze_stage60_deployable_gate.py `
    --neural $Neural --counts $Counts --output "$Run/diagnostic.json"
if ($LASTEXITCODE -ne 0) { throw "Stage60 diagnostic failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage60 complete: validation cross-fit diagnostic only; no export or test scoring."
