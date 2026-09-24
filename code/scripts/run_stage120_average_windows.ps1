$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage120-balanced-hard-continuation-s120017"
$Output = "$Run/average-validation.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage120 average score" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/score_stage120_average.py `
    --neural "$Run/average.pt" `
    --counts "runs/stage73-order6-b/counts/checkpoint.pt" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage120 average validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
