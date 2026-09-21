$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Authorized only after Stage 4A passed its predeclared quality, CPU time, RAM,
# and asset-size gates. This is a fresh 4,800-step run, not a resumed short run.
$RunDir = "runs/stage4-long-scaled-s17"
if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }

Write-Output "Starting $RunDir"
& $Python train_experiment.py `
    --implementation student --config configs/student_scaled.json `
    --device cuda --precision auto --seed 17 `
    --steps 4800 --micro-batch-size 32 --grad-accum 1 `
    --schedule baseline --eval-every 600 --save-every 200 --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunDir" }

& $Python evaluate.py --checkpoint "$RunDir/checkpoint.pt" `
    --device cpu --precision fp32 --split validation `
    --output "$RunDir/validation_cpu_fp32.json"
if ($LASTEXITCODE -ne 0) { throw "CPU validation failed: $RunDir" }

& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Output "Stage 4B complete. Validation only; no test evaluation."
