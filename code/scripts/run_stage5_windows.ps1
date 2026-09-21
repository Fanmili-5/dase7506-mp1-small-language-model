$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Stage 5A capacity-shape screen, fixed before observing any result. All three
# candidates start from random initialization and use validation only.
$Experiments = @(
    @{ Name = "w192_d8-s17"; Config = "configs/student_w192_d8.json" },
    @{ Name = "w224_d6-s17"; Config = "configs/student_w224_d6.json" },
    @{ Name = "w256_d6-s17"; Config = "configs/student_w256_d6.json" }
)

foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage5-$($Experiment.Name)"
    if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }
}

foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage5-$($Experiment.Name)"
    Write-Output "Starting $RunDir"
    & $Python train_experiment.py `
        --implementation student --config $Experiment.Config `
        --device cuda --precision auto --seed 17 `
        --steps 1200 --micro-batch-size 32 --grad-accum 1 `
        --schedule baseline --eval-every 300 --save-every 100 --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunDir" }
    & $Python evaluate.py --checkpoint "$RunDir/checkpoint.pt" `
        --device cpu --precision fp32 --split validation `
        --output "$RunDir/validation_cpu_fp32.json"
    if ($LASTEXITCODE -ne 0) { throw "CPU validation failed: $RunDir" }
}

& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Output "Stage 5A complete. Validation only; no test evaluation."
