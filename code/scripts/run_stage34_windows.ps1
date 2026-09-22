$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage34-depth10-width224-s17"
$PreflightPath = "runs/stage33-depth-preflight/preflight.json"
if (Test-Path $Run) { throw "Refusing to overwrite Stage34" }
if (-not (Test-Path $PreflightPath)) { throw "Stage33 preflight is missing" }
$Preflight = Get-Content $PreflightPath -Raw | ConvertFrom-Json
if (-not $Preflight.pass) { throw "Stage33 resource preflight did not pass" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_multi_token -v
if ($LASTEXITCODE -ne 0) { throw "Multi-token tests failed" }
& $Python scripts/train_stage34_depth_multi_token.py --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage34 training failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-006000.pt" `
    --checkpoint "$Run/checkpoints/step-006300.pt" `
    --checkpoint "$Run/checkpoints/step-006600.pt" `
    --checkpoint "$Run/checkpoints/step-006900.pt" `
    --checkpoint "$Run/checkpoints/step-007200.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage34 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_multi_token.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage34 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage34 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage34 complete: matched-target 10x224 multi-token training; no test scoring."
