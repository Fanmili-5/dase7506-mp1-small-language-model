$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage99-distilled-order6.json"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Base = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Extended = "runs/stage73-order6-b/counts/checkpoint.pt"
if (Test-Path $Output) { throw "Refusing to overwrite Stage99" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/scan_stage99_distilled_order6.py `
    --neural $Neural --base $Base --extended $Extended --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage99 distilled order-6 scan failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage99 complete: validation-only order-6 scan; no test scoring."
