$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage57-continuation-mkn-scan"
$Neural = "runs/stage56-hybrid-conv-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$NeuralSha = "b1cb9b8f7b8d94e9b8961f446b98bd80428e73818aceb4733e7af05fbaa2146b"
if (Test-Path $Run) { throw "Refusing to overwrite Stage57" }
if (-not (Test-Path $Neural)) { throw "Stage56 exported average is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Neural --counts $Counts --baseline $Baseline --run-dir $Run `
    --expected-neural-sha $NeuralSha --source-stage Stage57
if ($LASTEXITCODE -ne 0) { throw "Stage57 scan or qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage57 complete: inspect continuation/MKN qualification; no test scoring."
