$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage73-order6-b/counts/checkpoint.pt"
$Output = "runs/stage112-block-ablation.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage112" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage112_block_ablation.py `
    --neural $Neural --counts $Counts --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage112 diagnostic failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage112 diagnostic completed; no export or test scoring."
