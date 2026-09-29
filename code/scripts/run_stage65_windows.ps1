$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage65-hybrid-conv-byte-rdrop-s17"
$Start = "runs/stage63-primary-emphasis-s17/average-training.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage65" }
if (-not (Test-Path $Start)) { throw "Stage63 averaged training checkpoint is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv_byte_rdrop -v
if ($LASTEXITCODE -ne 0) { throw "Hybrid-conv byte/R-Drop tests failed" }
& $Python scripts/train_stage65_hybrid_conv_byte_rdrop.py --start $Start --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage65 continuation failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --checkpoint "$Run/checkpoints/step-002700.pt" `
    --checkpoint "$Run/checkpoints/step-003000.pt" `
    --checkpoint "$Run/checkpoints/step-003300.pt" `
    --checkpoint "$Run/checkpoints/step-003600.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage65 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_stage65_hybrid_conv_byte_rdrop.py `
    --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage65 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage65 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage65 complete: byte-composed hybrid-conv R-Drop continuation; no test scoring."
