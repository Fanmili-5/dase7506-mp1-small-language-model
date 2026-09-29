$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage20-fast-inference"
$Source = "runs/stage19-complementarity/mixture/best-mixture.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage20" }
if ((Get-FileHash $Baseline -Algorithm SHA256).Hash.ToLower() -ne "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d") { throw "Baseline changed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_ngram_fast
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
& $Python scripts/prepare_stage20_fast.py --source $Source --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Full validation equivalence failed" }
& $Python evaluate.py --checkpoint "$Run/fast-hybrid.pt" --device cpu --precision fp32 --threads 4 --split validation --output "$Run/validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Independent official validation failed" }
& $Python scripts/benchmark_cpu.py --baseline $Baseline --candidate "$Run/fast-hybrid.pt" --repeats 3 --threads 4 --output "$Run/resources.json"
if ($LASTEXITCODE -ne 0) { throw "Resource measurement failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed-file check failed" }
Write-Output "Stage20 completed; inspect resource decision. No training or test scoring."
