$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage39-fast-calibrated"
$Source = "runs/stage37-calibrated-qualified/calibrated-collapsed.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage39" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_calibrated_collapsed -v
if ($LASTEXITCODE -ne 0) { throw "Fast calibrated tests failed" }
& $Python scripts/prepare_stage39_fast_calibrated.py --source $Source --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage39 equivalence failed" }
& $Python evaluate.py --checkpoint "$Run/calibrated-collapsed.pt" --device cpu `
    --precision fp32 --threads 4 --split validation --output "$Run/validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage39 independent validation failed" }
& $Python scripts/benchmark_cpu.py --baseline $Baseline --candidate "$Run/calibrated-collapsed.pt" `
    --repeats 3 --threads 4 --output "$Run/resources.json"
if ($LASTEXITCODE -ne 0) { throw "Stage39 resources failed" }
& $Python scripts/finalize_stage37.py --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage39 final audit failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage39 complete: inspect fast calibrated qualification; no test scoring."
