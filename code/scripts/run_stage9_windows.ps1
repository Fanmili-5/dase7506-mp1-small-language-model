$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Frozen Stage-8 recipe, replicated at two new seeds. These runs must not alter
# architecture, optimizer, schedule, step count, or checkpoint-selection rule.
foreach ($Seed in @(23, 42)) {
    $RunDir = "runs/stage9-long-drop010-w256_d6-s$Seed"
    if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }

    & $Python train_experiment.py `
        --implementation student --config configs/student_w256_d6_drop010.json `
        --device cuda --precision auto --seed $Seed `
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
}

& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Output "Stage 9A complete. Validation only; no test evaluation."
