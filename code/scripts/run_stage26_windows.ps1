$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage26-multi-token-s17"
if (Test-Path $Run) { throw "Refusing to overwrite Stage26" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_multi_token -v
if ($LASTEXITCODE -ne 0) { throw "Multi-token tests failed" }
& $Python scripts/train_stage26_multi_token.py --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage26 training failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-006000.pt" `
    --checkpoint "$Run/checkpoints/step-006300.pt" `
    --checkpoint "$Run/checkpoints/step-006600.pt" `
    --checkpoint "$Run/checkpoints/step-006900.pt" `
    --checkpoint "$Run/checkpoints/step-007200.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage26 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_multi_token.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage26 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage26 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage26 complete: training-only multi-token prediction; no test scoring."
