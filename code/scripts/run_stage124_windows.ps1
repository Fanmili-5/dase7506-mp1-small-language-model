$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage124-torchscript-features.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage124" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/probe_stage124_torchscript_features.py `
    --neural "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage124 CPU graph pilot failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
