$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Fresh duration screen for the Stage-6 winner. The 4,800-step cosine plan is
# fixed from initialization; no 3,600-step state is resumed.
$RunDir = "runs/stage8-long-drop010-w256_d6-s17"
if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }

& $Python train_experiment.py `
    --implementation student --config configs/student_w256_d6_drop010.json `
    --device cuda --precision auto --seed 17 `
    --steps 4800 --micro-batch-size 32 --grad-accum 1 `
    --schedule baseline --eval-every 300 --save-every 200 --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunDir" }

& $Python evaluate.py --checkpoint "$RunDir/checkpoint.pt" `
    --device cpu --precision fp32 --split validation `
    --output "$RunDir/validation_cpu_fp32.json"
if ($LASTEXITCODE -ne 0) { throw "CPU endpoint validation failed: $RunDir" }

& $Python evaluate.py --checkpoint "$RunDir/checkpoint-best.pt" `
    --device cpu --precision fp32 --split validation `
    --output "$RunDir/validation_best_cpu_fp32.json"
if ($LASTEXITCODE -ne 0) { throw "CPU best-checkpoint validation failed: $RunDir" }

& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Output "Stage 8A complete. Validation only; no test evaluation."
