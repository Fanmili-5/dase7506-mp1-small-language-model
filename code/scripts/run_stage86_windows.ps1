$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage86-calibration-aware-s86017"
$Start = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage86" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_mixture_aware tests.test_calibrated_hybrid_conv -v
if ($LASTEXITCODE -ne 0) { throw "Calibration-aware dependency tests failed" }
& $Python scripts/train_stage86_calibration_aware.py `
    --start $Start --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage86 training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-001200.pt" `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --checkpoint "$Run/checkpoints/step-002100.pt" `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage86 averaging failed" }
& $Python scripts/score_stage86_average.py `
    --neural "$Run/average.pt" --counts $Counts `
    --output "$Run/average-calibrated-validation.json"
if ($LASTEXITCODE -ne 0) { throw "Stage86 average scoring failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage86 complete: inspect fixed calibrated average; no test scoring."
