$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Neural = "runs/stage74-balanced-hybrid-rdrop-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$Run = "runs/stage75-balanced-hybrid-mkn"
if (Test-Path $Run) { throw "Refusing to overwrite Stage75" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
$NeuralSha = (Get-FileHash $Neural -Algorithm SHA256).Hash.ToLower()
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Neural --counts $Counts --baseline $Baseline `
    --run-dir $Run --expected-neural-sha $NeuralSha --source-stage Stage74
if ($LASTEXITCODE -ne 0) { throw "Stage75 MKN qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage75 complete: balanced-hybrid MKN scan and qualification; no test scoring."
