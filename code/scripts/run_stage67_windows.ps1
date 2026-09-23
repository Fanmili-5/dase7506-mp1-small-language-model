$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage67-output-bias-s17"
$Start = "runs/stage65-hybrid-conv-byte-rdrop-s17/average-inference.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage67" }
if (-not (Test-Path $Start)) { throw "Stage65 exported average is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_hybrid_conv_output_bias -v
if ($LASTEXITCODE -ne 0) { throw "Output-bias tests failed" }
& $Python scripts/fit_stage67_output_bias.py --start $Start --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage67 output-bias fit failed" }
$Average = "$Run/average.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/epoch-03.pt" `
    --checkpoint "$Run/checkpoints/epoch-04.pt" `
    --checkpoint "$Run/checkpoints/epoch-05.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage67 averaging failed" }
& $Python evaluate.py --checkpoint $Average --device cpu --precision fp32 --threads 4 `
    --split validation --output "$Run/average-validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage67 independent validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage67 complete: train-only vocabulary intercept fit; no test scoring."
