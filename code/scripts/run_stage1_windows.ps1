param(
    [int]$Seed = 17,
    [int]$Steps = 1200,
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run scripts/setup_windows_cuda.ps1 first."
}

$Experiments = @(
    @{ Name = "baseline"; Implementation = "model"; Config = "configs/baseline.json" },
    @{ Name = "rope"; Implementation = "student"; Config = "configs/student_rope.json" },
    @{ Name = "swiglu"; Implementation = "student"; Config = "configs/student_swiglu.json" },
    @{ Name = "modern"; Implementation = "student"; Config = "configs/student_modern.json" }
)

foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage1-$($Experiment.Name)-s$Seed"
    if (Test-Path $RunDir) {
        throw "Run directory already exists: $RunDir. Preserve it and choose another seed/name."
    }
    & $Python train_experiment.py `
        --implementation $Experiment.Implementation `
        --config $Experiment.Config `
        --device cuda `
        --precision auto `
        --seed $Seed `
        --steps $Steps `
        --micro-batch-size $BatchSize `
        --grad-accum 1 `
        --schedule baseline `
        --eval-every 300 `
        --save-every 100 `
        --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunDir" }

    & $Python evaluate.py `
        --checkpoint "$RunDir/checkpoint.pt" `
        --device cpu `
        --precision fp32 `
        --split validation `
        --output "$RunDir/validation_cpu_fp32.json"
    if ($LASTEXITCODE -ne 0) { throw "CPU FP32 validation failed: $RunDir" }
}

& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Host "Stage 1 complete. Review results/summary.csv before any test evaluation." -ForegroundColor Green
