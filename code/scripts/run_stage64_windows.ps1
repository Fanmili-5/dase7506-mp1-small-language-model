$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage64-primary-emphasis-mkn-scan"
$Neural = "runs/stage63-primary-emphasis-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$NeuralSha = "878c728f43c9e2d359e165b2dfdc1e3baeed79efa4bee7e33696906378dc0b25"
if (Test-Path $Run) { throw "Refusing to overwrite Stage64" }
if (-not (Test-Path $Neural)) { throw "Stage63 exported average is required" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Neural --counts $Counts --baseline $Baseline --run-dir $Run `
    --expected-neural-sha $NeuralSha --source-stage Stage64
if ($LASTEXITCODE -ne 0) { throw "Stage64 scan or qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage64 complete: inspect primary-emphasis/MKN qualification; no test scoring."
