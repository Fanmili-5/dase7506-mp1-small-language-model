$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage74-balanced-hybrid-rdrop-s17"
if (Test-Path $Run) { throw "Refusing to overwrite Stage74" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv tests.test_hybrid_conv_rdrop -v
if ($LASTEXITCODE -ne 0) { throw "Balanced hybrid R-Drop tests failed" }
& $Python scripts/train_stage74_balanced_hybrid_rdrop.py --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage74 training failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-006000.pt" `
    --checkpoint "$Run/checkpoints/step-006300.pt" `
    --checkpoint "$Run/checkpoints/step-006600.pt" `
    --checkpoint "$Run/checkpoints/step-006900.pt" `
    --checkpoint "$Run/checkpoints/step-007200.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage74 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_stage54_hybrid_conv_rdrop.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage74 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage74 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage74 complete: admitted balanced-hybrid R-Drop quality run; no test scoring."
