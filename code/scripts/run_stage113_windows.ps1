$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage113-pruned-block5-repair-s113017"
$StudentStart = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage73-order6-b/counts/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage113" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage113_pruned_block_repair.py `
    --student-start $StudentStart --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage113 repair training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --checkpoint "$Run/checkpoints/step-002100.pt" `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage113 averaging failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage113 completed; inspect validation before export or test scoring."
