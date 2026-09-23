$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage59-local-heavy-rdrop-s17"
if (Test-Path $Run) { throw "Refusing to overwrite Stage59" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv tests.test_hybrid_conv_rdrop -v
if ($LASTEXITCODE -ne 0) { throw "Local-heavy R-Drop tests failed" }
& $Python scripts/train_stage59_local_heavy_rdrop.py --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage59 training failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-006000.pt" `
    --checkpoint "$Run/checkpoints/step-006300.pt" `
    --checkpoint "$Run/checkpoints/step-006600.pt" `
    --checkpoint "$Run/checkpoints/step-006900.pt" `
    --checkpoint "$Run/checkpoints/step-007200.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage59 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_stage54_hybrid_conv_rdrop.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage59 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage59 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage59 complete: local-heavy R-Drop training; no test scoring."
