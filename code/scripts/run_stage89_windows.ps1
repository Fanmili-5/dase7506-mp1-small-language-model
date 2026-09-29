$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage89-calibrated-gate.json"
$Neural = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Output) { throw "Refusing to overwrite Stage89" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/analyze_stage89_calibrated_gate.py `
    --neural $Neural --counts $Counts --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage89 calibrated-gate diagnostic failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage89 complete: diagnostic only; no export and no test scoring."
