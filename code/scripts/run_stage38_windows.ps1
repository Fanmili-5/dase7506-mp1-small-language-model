$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage38-adam-beta95-s17"
if (Test-Path $Run) { throw "Refusing to overwrite Stage38" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_multi_token -v
if ($LASTEXITCODE -ne 0) { throw "Multi-token tests failed" }
& $Python scripts/train_stage34_depth_multi_token.py --run-dir $Run `
    --config configs/stage26_multi_token.json --beta2 0.95
if ($LASTEXITCODE -ne 0) { throw "Stage38 training failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-006000.pt" `
    --checkpoint "$Run/checkpoints/step-006300.pt" `
    --checkpoint "$Run/checkpoints/step-006600.pt" `
    --checkpoint "$Run/checkpoints/step-006900.pt" `
    --checkpoint "$Run/checkpoints/step-007200.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage38 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_multi_token.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage38 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage38 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage38 complete: beta2=0.95 matched optimizer ablation; no test scoring."
