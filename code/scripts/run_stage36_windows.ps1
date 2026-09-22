$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage36-depth9-width240-s17"
$PreflightPath = "runs/stage35-depth9-preflight/preflight.json"
if (Test-Path $Run) { throw "Refusing to overwrite Stage36" }
if (-not (Test-Path $PreflightPath)) { throw "Stage35 preflight is missing" }
$Preflight = Get-Content $PreflightPath -Raw | ConvertFrom-Json
if (-not $Preflight.pass) { throw "Stage35 resource preflight did not pass" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_multi_token -v
if ($LASTEXITCODE -ne 0) { throw "Multi-token tests failed" }
& $Python scripts/train_stage34_depth_multi_token.py --run-dir $Run `
    --config configs/stage35_depth9_width240.json
if ($LASTEXITCODE -ne 0) { throw "Stage36 training failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-006000.pt" `
    --checkpoint "$Run/checkpoints/step-006300.pt" `
    --checkpoint "$Run/checkpoints/step-006600.pt" `
    --checkpoint "$Run/checkpoints/step-006900.pt" `
    --checkpoint "$Run/checkpoints/step-007200.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage36 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_multi_token.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage36 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage36 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage36 complete: matched-target 9x240 multi-token training; no test scoring."
