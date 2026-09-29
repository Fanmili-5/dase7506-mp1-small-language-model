$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage98-distilled-gate.json"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Output) { throw "Refusing to overwrite Stage98" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/analyze_stage98_distilled_gate.py `
    --neural $Neural --counts $Counts --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage98 distilled gate failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage98 complete: validation-only cross-fit; no export or test scoring."
