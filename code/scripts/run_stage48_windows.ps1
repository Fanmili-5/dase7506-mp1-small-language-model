$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage48-top1-moe-s17"
$PreflightPath = "runs/stage46-top1-moe-preflight/preflight.json"
if (Test-Path $Run) { throw "Refusing to overwrite Stage48" }
if (-not (Test-Path $PreflightPath)) { throw "Stage46 preflight is missing" }
$Preflight = Get-Content $PreflightPath -Raw | ConvertFrom-Json
if (-not $Preflight.pass) { throw "Stage46 resource preflight did not pass" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_moe -v
if ($LASTEXITCODE -ne 0) { throw "MoE tests failed" }
& $Python scripts/train_stage48_moe.py --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage48 training failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-006000.pt" `
    --checkpoint "$Run/checkpoints/step-006300.pt" `
    --checkpoint "$Run/checkpoints/step-006600.pt" `
    --checkpoint "$Run/checkpoints/step-006900.pt" `
    --checkpoint "$Run/checkpoints/step-007200.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage48 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_stage48_moe.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage48 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage48 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage48 complete: matched-target top-1 MoE training; no test scoring."
