$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage120-balanced-hard-continuation-s120017"
$Student = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Primary = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Alternate = "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt"
$Counts = "runs/stage73-order6-b/counts/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage120" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage120_hard_balanced_continuation.py `
    --student-start $Student --primary $Primary --alternate $Alternate `
    --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage120 continuation failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --checkpoint "$Run/checkpoints/step-003000.pt" `
    --checkpoint "$Run/checkpoints/step-003600.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage120 averaging failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage120 complete: score average before any export or test scoring."
