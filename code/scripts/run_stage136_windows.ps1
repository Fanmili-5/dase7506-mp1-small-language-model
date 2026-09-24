$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage136-successor-head-s136017"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage136" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_stage135_successor_gate `
    tests.test_stage136_successor_neural -q
if ($LASTEXITCODE -ne 0) { throw "Successor-head tests failed" }
& $Python scripts/train_stage136_successor_head.py `
    --neural $Neural --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage136 training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --checkpoint "$Run/checkpoints/step-002100.pt" `
    --checkpoint "$Run/checkpoints/step-002400.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage136 fixed average failed" }
& $Python scripts/score_stage136_average.py `
    --neural "$Run/average.pt" --counts $Counts `
    --output "$Run/average-validation.json"
if ($LASTEXITCODE -ne 0) { throw "Stage136 average validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage136 train-only successor-head experiment complete; no test scoring."
