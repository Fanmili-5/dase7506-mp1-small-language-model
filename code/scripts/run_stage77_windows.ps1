$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Neural = "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$Run = "runs/stage77-balanced-continuation-mkn"
if (Test-Path $Run) { throw "Refusing to overwrite Stage77" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
$NeuralSha = (Get-FileHash $Neural -Algorithm SHA256).Hash.ToLower()
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Neural --counts $Counts --baseline $Baseline `
    --run-dir $Run --expected-neural-sha $NeuralSha --source-stage Stage76
if ($LASTEXITCODE -ne 0) { throw "Stage77 MKN qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage77 complete: balanced-continuation MKN qualification; no test scoring."
