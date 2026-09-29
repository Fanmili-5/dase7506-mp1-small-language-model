$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage92-heterogeneous-distillation-s92017"
$Primary = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Alternate = "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage92" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_mixture_aware tests.test_calibrated_hybrid_conv `
    tests.test_hybrid_conv -v
if ($LASTEXITCODE -ne 0) { throw "Stage92 dependency tests failed" }
& $Python scripts/train_stage92_heterogeneous_distillation.py `
    --primary $Primary --alternate $Alternate --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage92 training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-000900.pt" `
    --checkpoint "$Run/checkpoints/step-001200.pt" `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage92 averaging failed" }
& $Python scripts/score_stage92_average.py `
    --neural "$Run/average.pt" --counts $Counts `
    --output "$Run/average-calibrated-validation.json"
if ($LASTEXITCODE -ne 0) { throw "Stage92 average scoring failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage92 complete: inspect fixed distilled average; no test scoring."
