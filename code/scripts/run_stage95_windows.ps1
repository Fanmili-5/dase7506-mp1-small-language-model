$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage95-balanced-student-distillation-s95017"
$Primary = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Alternate = "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage95" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage95_balanced_student_distillation.py `
    --primary $Primary --alternate $Alternate --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage95 training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-001200.pt" `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --checkpoint "$Run/checkpoints/step-002100.pt" `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage95 averaging failed" }
& $Python scripts/score_stage92_average.py `
    --neural "$Run/average.pt" --counts $Counts `
    --output "$Run/average-calibrated-validation.json"
if ($LASTEXITCODE -ne 0) { throw "Stage95 average scoring failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage95 complete: inspect balanced distilled average; no test scoring."
