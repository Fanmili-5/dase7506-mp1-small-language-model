$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage62-second-continuation-mkn-scan"
$Neural = "runs/stage61-hybrid-conv-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$NeuralSha = "a6cd2f273837fe3c14545947ad8026de38624e889e07b54bf3698952dd6103c4"
if (Test-Path $Run) { throw "Refusing to overwrite Stage62" }
if (-not (Test-Path $Neural)) { throw "Stage61 exported average is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Neural --counts $Counts --baseline $Baseline --run-dir $Run `
    --expected-neural-sha $NeuralSha --source-stage Stage62
if ($LASTEXITCODE -ne 0) { throw "Stage62 scan or qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage62 complete: inspect second-continuation/MKN qualification; no test scoring."
