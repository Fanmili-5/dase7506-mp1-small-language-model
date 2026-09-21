$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Stage 6A is a matched 3,600-step regularization screen. The schedule length,
# seed, optimizer, target count, and validation cadence are identical across all
# three runs; only training dropout differs.
$Experiments = @(
    @{ Name = "schedule3600-w256_d6-s17"; Config = "configs/student_w256_d6.json" },
    @{ Name = "drop005-w256_d6-s17"; Config = "configs/student_w256_d6_drop005.json" },
    @{ Name = "drop010-w256_d6-s17"; Config = "configs/student_w256_d6_drop010.json" }
)

foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage6-$($Experiment.Name)"
    if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }
}

foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage6-$($Experiment.Name)"
    Write-Output "Starting $RunDir"
    & $Python train_experiment.py `
        --implementation student --config $Experiment.Config `
        --device cuda --precision auto --seed 17 `
        --steps 3600 --micro-batch-size 32 --grad-accum 1 `
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
Write-Output "Stage 6A complete. Validation only; no test evaluation."
