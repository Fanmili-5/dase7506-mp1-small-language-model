$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Stage 4A is a quality-and-resource gate, fixed before observing the result.
# The candidate trains from random initialization and is selected only on the
# complete validation split. The test split is not evaluated.
$RunDir = "runs/stage4-scaled-s17"
$BaselineCheckpoint = "runs/stage3-long-baseline-s17/checkpoint.pt"
$ResourceOutput = "results/cpu-stage4-scaled.json"
if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }
if (-not (Test-Path $BaselineCheckpoint)) {
    throw "Missing resource-control checkpoint: $BaselineCheckpoint"
}
if (Test-Path $ResourceOutput) { throw "Resource output already exists: $ResourceOutput" }

Write-Output "Starting $RunDir"
& $Python train_experiment.py `
    --implementation student --config configs/student_scaled.json `
    --device cuda --precision auto --seed 17 `
    --steps 1200 --micro-batch-size 32 --grad-accum 1 `
    --schedule baseline --eval-every 300 --save-every 100 --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunDir" }

& $Python evaluate.py --checkpoint "$RunDir/checkpoint.pt" `
    --device cpu --precision fp32 --split validation `
    --output "$RunDir/validation_cpu_fp32.json"
if ($LASTEXITCODE -ne 0) { throw "CPU validation failed: $RunDir" }

& $Python scripts/benchmark_cpu.py `
    --baseline $BaselineCheckpoint --candidate "$RunDir/checkpoint.pt" `
    --repeats 3 --threads 4 --output $ResourceOutput
if ($LASTEXITCODE -ne 0) { throw "CPU resource gate failed: $RunDir" }

& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Output "Stage 4A complete. Validation only; no test evaluation."
