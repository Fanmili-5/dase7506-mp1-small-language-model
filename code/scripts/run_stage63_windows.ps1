$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage63-primary-emphasis-s17"
$Start = "runs/stage61-hybrid-conv-continuation-s17/average-training.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage63" }
if (-not (Test-Path $Start)) { throw "Stage61 averaged training checkpoint is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage63_primary_emphasis.py --start $Start --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage63 continuation failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --checkpoint "$Run/checkpoints/step-002700.pt" `
    --checkpoint "$Run/checkpoints/step-003000.pt" `
    --checkpoint "$Run/checkpoints/step-003300.pt" `
    --checkpoint "$Run/checkpoints/step-003600.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage63 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_stage54_hybrid_conv_rdrop.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage63 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage63 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage63 complete: primary-emphasis continuation and fixed averaging; no test scoring."
