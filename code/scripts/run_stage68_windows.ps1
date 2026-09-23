$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage68-output-bias-mkn-scan"
$Neural = "runs/stage67-output-bias-s17/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$NeuralSha = "3be9468b122da4c486726e8bcc59692005d0fc90dcbdd6a59c41b24d42d20b0f"
if (Test-Path $Run) { throw "Refusing to overwrite Stage68" }
if (-not (Test-Path $Neural)) { throw "Stage67 averaged checkpoint is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_ngram_hybrid_conv_bias_collapsed -v
if ($LASTEXITCODE -ne 0) { throw "Output-bias collapsed-mixture tests failed" }
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Neural --counts $Counts --baseline $Baseline --run-dir $Run `
    --expected-neural-sha $NeuralSha --source-stage Stage68
if ($LASTEXITCODE -ne 0) { throw "Stage68 scan or qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage68 complete: inspect output-bias/MKN qualification; no test scoring."
