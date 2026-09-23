$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage69-output-lora-s17"
$Start = "runs/stage67-output-bias-s17/average.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage69" }
if (-not (Test-Path $Start)) { throw "Stage67 averaged checkpoint is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv_output_lora -v
if ($LASTEXITCODE -ne 0) { throw "Output-LoRA tests failed" }
& $Python scripts/fit_stage69_output_lora.py --start $Start --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage69 output-LoRA fit failed" }
$Average = "$Run/average-training.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/epoch-03.pt" `
    --checkpoint "$Run/checkpoints/epoch-04.pt" `
    --checkpoint "$Run/checkpoints/epoch-05.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage69 averaging failed" }
$Exported = "$Run/average-inference.pt"
& $Python scripts/export_stage69_output_lora.py --checkpoint $Average --output $Exported
if ($LASTEXITCODE -ne 0) { throw "Stage69 export failed" }
& $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage69 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage69 complete: folded output-LoRA fit; no test scoring."
