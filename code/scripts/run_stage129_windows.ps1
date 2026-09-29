$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage129-merged-backbone-lora-s129017"
if (Test-Path $Run) { throw "Refusing to overwrite Stage129" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage129_backbone_lora.py `
    --student "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --primary "runs/stage71-mixture-aware-s71017-d/average.pt" `
    --alternate "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt" `
    --counts "runs/stage25-kneser-ney/counts-min2/checkpoint.pt" `
    --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage129 training failed" }
& $Python scripts/average_checkpoints.py `
    --checkpoint "$Run/checkpoints/step-000900.pt" `
    --checkpoint "$Run/checkpoints/step-001200.pt" `
    --checkpoint "$Run/checkpoints/step-001500.pt" `
    --checkpoint "$Run/checkpoints/step-001800.pt" `
    --output "$Run/average.pt"
if ($LASTEXITCODE -ne 0) { throw "Stage129 checkpoint averaging failed" }
& $Python scripts/score_stage129_average.py `
    --neural "$Run/average.pt" `
    --counts "runs/stage25-kneser-ney/counts-min2/checkpoint.pt" `
    --output "$Run/average-validation.json"
if ($LASTEXITCODE -ne 0) { throw "Stage129 average validation failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage129 complete: inspect quality before export; no test scoring."
