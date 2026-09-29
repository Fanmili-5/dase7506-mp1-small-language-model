$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage76-balanced-hybrid-continuation-s17"
$Start = "runs/stage74-balanced-hybrid-rdrop-s17/average-training.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage76" }
if (-not (Test-Path $Start)) { throw "Stage74 averaged training checkpoint is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv tests.test_hybrid_conv_rdrop -v
if ($LASTEXITCODE -ne 0) { throw "Balanced hybrid continuation tests failed" }
& $Python scripts/train_stage76_balanced_hybrid_continuation.py --start $Start --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage76 continuation failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-003600.pt" `
    --checkpoint "$Run/checkpoints/step-003900.pt" `
    --checkpoint "$Run/checkpoints/step-004200.pt" `
    --checkpoint "$Run/checkpoints/step-004500.pt" `
    --checkpoint "$Run/checkpoints/step-004800.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage76 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_stage54_hybrid_conv_rdrop.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage76 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage76 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage76 complete: balanced-hybrid low-LR continuation; no test scoring."
