$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage66-byte-composed-mkn-scan"
$Neural = "runs/stage65-hybrid-conv-byte-rdrop-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$NeuralSha = "f9c8b41efea91d5594b25760c91652edc071a161983b12b22cc738988a7a9bdc"
if (Test-Path $Run) { throw "Refusing to overwrite Stage66" }
if (-not (Test-Path $Neural)) { throw "Stage65 exported average is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Neural --counts $Counts --baseline $Baseline --run-dir $Run `
    --expected-neural-sha $NeuralSha --source-stage Stage66
if ($LASTEXITCODE -ne 0) { throw "Stage66 scan or qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage66 complete: inspect byte-composed/MKN qualification; no test scoring."
