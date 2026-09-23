$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage71-mixture-aware-s71017"
$Start = "runs/stage67-output-bias-s17/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage71" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_mixture_aware -v
if ($LASTEXITCODE -ne 0) { throw "Mixture-aware tests failed" }
& $Python scripts/train_stage71_mixture_aware.py `
    --start $Start --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage71 mixture-aware continuation failed" }
$Average = "$Run/average.pt"
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --checkpoint "$Run/checkpoints/step-002700.pt" `
    --checkpoint "$Run/checkpoints/step-003000.pt" `
    --checkpoint "$Run/checkpoints/step-003300.pt" `
    --checkpoint "$Run/checkpoints/step-003600.pt" `
    --output $Average
if ($LASTEXITCODE -ne 0) { throw "Stage71 averaging failed" }
$NeuralSha = (Get-FileHash $Average -Algorithm SHA256).Hash.ToLower()
$Qualification = "runs/stage71-mixture-aware-mkn"
& $Python scripts/prepare_stage55_hybrid_conv_mkn.py `
    --neural $Average --counts $Counts --baseline $Baseline `
    --run-dir $Qualification --expected-neural-sha $NeuralSha `
    --source-stage Stage71
if ($LASTEXITCODE -ne 0) { throw "Stage71 MKN qualification failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage71 complete: inspect mixture-aware quality and qualification; no test scoring."
