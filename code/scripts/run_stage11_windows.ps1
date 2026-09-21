$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

$Primary = "runs/stage8-long-drop010-w256_d6-s17/checkpoint-best.pt"
$Auxiliary = "runs/stage3-long-modern-s17/checkpoint-best.pt"
$RunDirectory = "runs/stage11-ensemble-w080-s17"
$Checkpoint = "$RunDirectory/checkpoint.pt"
$Validation = "$RunDirectory/validation_cpu_fp32.json"
$Resource = "results/cpu-stage11-ensemble-w080-s17.json"
foreach ($Path in @($Primary, $Auxiliary)) {
    if (-not (Test-Path $Path)) { throw "Missing source checkpoint: $Path" }
}
foreach ($Path in @($Checkpoint, $Validation, $Resource)) {
    if (Test-Path $Path) { throw "Refusing to overwrite existing output: $Path" }
}

& $Python scripts/make_ensemble_checkpoint.py `
    --primary $Primary --auxiliary $Auxiliary --primary-weight 0.8 --output $Checkpoint
if ($LASTEXITCODE -ne 0) { throw "Ensemble packaging failed." }
& $Python evaluate.py --checkpoint $Checkpoint --device cpu --precision fp32 `
    --threads 4 --split validation --output $Validation
if ($LASTEXITCODE -ne 0) { throw "Ensemble validation failed." }
& $Python scripts/benchmark_cpu.py --baseline runs/stage3-long-baseline-s17/checkpoint-best.pt `
    --candidate $Checkpoint --repeats 3 --threads 4 --output $Resource
if ($LASTEXITCODE -ne 0) { throw "Ensemble resource benchmark failed." }
Write-Output "Stage 11 complete. Validation and resource checks only; no test evaluation."
