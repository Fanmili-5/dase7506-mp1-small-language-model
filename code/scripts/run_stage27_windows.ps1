$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage27-mkn-collapsed"
$Source = "runs/stage25-kneser-ney/scan-min2/best-mixture.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage27" }
if ((Get-FileHash $Source -Algorithm SHA256).Hash.ToLower() -ne "3669552af47ef7cae72c14be15359d798ae1afbe7fbd3608f21d43414a555ad5") { throw "Stage25 source changed" }
if ((Get-FileHash $Baseline -Algorithm SHA256).Hash.ToLower() -ne "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d") { throw "Baseline changed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_kneser_ney tests.test_ngram_collapsed -v
if ($LASTEXITCODE -ne 0) { throw "Count/collapsed tests failed" }
& $Python scripts/prepare_stage27_mkn_collapsed.py --source $Source --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Full validation equivalence failed" }
& $Python evaluate.py --checkpoint "$Run/collapsed-hybrid.pt" --device cpu --precision fp32 --threads 4 --split validation --output "$Run/validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Independent official validation failed" }
& $Python scripts/benchmark_cpu.py --baseline $Baseline --candidate "$Run/collapsed-hybrid.pt" --repeats 3 --threads 4 --output "$Run/resources.json"
if ($LASTEXITCODE -ne 0) { throw "Resource measurement failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed-file check failed" }
Write-Output "Stage27 complete: inspect modified-KN exact resources; no test scoring."
