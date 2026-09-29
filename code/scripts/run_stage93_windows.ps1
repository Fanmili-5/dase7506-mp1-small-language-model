$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage93-pure-distillation-s92017"
$Start = "runs/stage92-heterogeneous-distillation-s92017/checkpoints/step-001800.pt"
$Primary = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Alternate = "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage93" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage93_pure_distillation.py `
    --start $Start --primary $Primary --alternate $Alternate `
    --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage93 training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-000600.pt" `
    --checkpoint "$Run/checkpoints/step-000900.pt" `
    --checkpoint "$Run/checkpoints/step-001200.pt" `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage93 averaging failed" }
& $Python scripts/score_stage93_average.py `
    --neural "$Run/average.pt" --counts $Counts `
    --output "$Run/average-calibrated-validation.json"
if ($LASTEXITCODE -ne 0) { throw "Stage93 average scoring failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage93 complete: inspect fixed pure-distillation average; no test scoring."
