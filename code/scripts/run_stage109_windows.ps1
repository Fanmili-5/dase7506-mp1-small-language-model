$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage109-teacher-only-continuation-s109017"
$Student = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Primary = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Alternate = "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage109" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage109_teacher_only_continuation.py `
    --student $Student --primary $Primary --alternate $Alternate `
    --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage109 training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --checkpoint "$Run/checkpoints/step-002100.pt" `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage109 averaging failed" }
& $Python scripts/score_stage92_average.py `
    --neural "$Run/average.pt" --counts $Counts `
    --output "$Run/average-calibrated-validation.json"
if ($LASTEXITCODE -ne 0) { throw "Stage109 average scoring failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage109 completed; inspect validation before any export or test scoring."
