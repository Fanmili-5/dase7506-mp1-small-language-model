$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage88-contiguous-soups.json"
$CheckpointDir = "runs/stage71-mixture-aware-s71017-d/checkpoints"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Output) { throw "Refusing to overwrite Stage88" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage88_contiguous_soups.py `
    --checkpoint-dir $CheckpointDir --counts $Counts --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage88 soup screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage88 complete: diagnostic only; no export and no test scoring."
