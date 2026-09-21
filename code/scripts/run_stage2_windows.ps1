$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Predeclared before looking at these results: six seed replications and a
# three-run conditional component ladder. Existing seed-17 controls are reused.
$Experiments = @()
foreach ($Seed in @(23, 42)) {
    foreach ($Name in @("baseline", "rope", "modern")) {
        $Implementation = "student"
        $Config = "configs/student_$Name.json"
        if ($Name -eq "baseline") {
            $Implementation = "model"
            $Config = "configs/baseline.json"
        }
        $Experiments += @{ Name = $Name; Seed = $Seed; Implementation = $Implementation; Config = $Config }
    }
}
foreach ($Name in @("rope_swiglu", "rope_swiglu_rms", "rope_swiglu_rms_nobias")) {
    $Experiments += @{ Name = $Name; Seed = 17; Implementation = "student"; Config = "configs/student_$Name.json" }
}

# Fail before starting if any destination already exists; never overwrite runs.
foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage2-$($Experiment.Name)-s$($Experiment.Seed)"
    if (Test-Path $RunDir) { throw "Run already exists: $RunDir" }
}
foreach ($Experiment in $Experiments) {
    $RunDir = "runs/stage2-$($Experiment.Name)-s$($Experiment.Seed)"
    Write-Output "Starting $RunDir"
    & $Python train_experiment.py `
        --implementation $Experiment.Implementation --config $Experiment.Config `
        --device cuda --precision auto --seed $Experiment.Seed `
        --steps 1200 --micro-batch-size 32 --grad-accum 1 --schedule baseline `
        --eval-every 300 --save-every 100 --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunDir" }
    & $Python evaluate.py --checkpoint "$RunDir/checkpoint.pt" `
        --device cpu --precision fp32 --split validation `
        --output "$RunDir/validation_cpu_fp32.json"
    if ($LASTEXITCODE -ne 0) { throw "CPU validation failed: $RunDir" }
}
& $Python scripts/collect_results.py
if ($LASTEXITCODE -ne 0) { throw "Result collection failed." }
Write-Output "Stage 2 complete. Validation only; no test evaluation."
