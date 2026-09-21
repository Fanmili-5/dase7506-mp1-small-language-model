$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Fixed before observing any stage-3 result. All runs start from random
# initialization; existing short checkpoints are comparison evidence only.
$Experiments = @(
    @{ Name = "rep-rope_swiglu-s23"; Implementation = "student"; Config = "configs/student_rope_swiglu.json"; Seed = 23; Steps = 1200; EvalEvery = 300; SaveEvery = 100 },
    @{ Name = "rep-rope_swiglu-s42"; Implementation = "student"; Config = "configs/student_rope_swiglu.json"; Seed = 42; Steps = 1200; EvalEvery = 300; SaveEvery = 100 },
    @{ Name = "long-baseline-s17"; Implementation = "model"; Config = "configs/baseline.json"; Seed = 17; Steps = 4800; EvalEvery = 600; SaveEvery = 200 },
    @{ Name = "long-modern-s17"; Implementation = "student"; Config = "configs/student_modern.json"; Seed = 17; Steps = 4800; EvalEvery = 600; SaveEvery = 200 },
    @{ Name = "long-rope_swiglu-s17"; Implementation = "student"; Config = "configs/student_rope_swiglu.json"; Seed = 17; Steps = 4800; EvalEvery = 600; SaveEvery = 200 }
)

foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage3-$($Experiment.Name)"
    if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }
}

foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage3-$($Experiment.Name)"
    Write-Output "Starting $RunDir"
    & $Python train_experiment.py `
        --implementation $Experiment.Implementation --config $Experiment.Config `
        --device cuda --precision auto --seed $Experiment.Seed `
        --steps $Experiment.Steps --micro-batch-size 32 --grad-accum 1 `
        --schedule baseline --eval-every $Experiment.EvalEvery `
        --save-every $Experiment.SaveEvery --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunDir" }
    & $Python evaluate.py --checkpoint "$RunDir/checkpoint.pt" `
        --device cpu --precision fp32 --split validation `
        --output "$RunDir/validation_cpu_fp32.json"
    if ($LASTEXITCODE -ne 0) { throw "CPU validation failed: $RunDir" }
}

& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Output "Stage 3 complete. Validation only; no test evaluation."
