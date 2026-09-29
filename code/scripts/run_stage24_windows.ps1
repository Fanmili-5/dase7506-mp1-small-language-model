$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage24-stage22-collapsed"
$Source = "runs/stage22-deep-supervision-hybrid-scan/best-mixture.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage24" }
if ((Get-FileHash $Source -Algorithm SHA256).Hash.ToLower() -ne "ffcd5a3b6c338f030c0c9019a9df8c0f05297e46a209c25a76ee8c237df1f6bd") { throw "Stage22 source changed" }
if ((Get-FileHash $Baseline -Algorithm SHA256).Hash.ToLower() -ne "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d") { throw "Baseline changed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_ngram_collapsed -v
if ($LASTEXITCODE -ne 0) { throw "Collapsed inference tests failed" }
& $Python scripts/prepare_stage24_collapsed.py --source $Source --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Full validation equivalence failed" }
& $Python evaluate.py --checkpoint "$Run/collapsed-hybrid.pt" --device cpu --precision fp32 --threads 4 --split validation --output "$Run/validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Independent official validation failed" }
& $Python scripts/benchmark_cpu.py --baseline $Baseline --candidate "$Run/collapsed-hybrid.pt" --repeats 3 --threads 4 --output "$Run/resources.json"
if ($LASTEXITCODE -ne 0) { throw "Resource measurement failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed-file check failed" }
Write-Output "Stage24 complete: inspect exact Stage22 resource decision; no test scoring."
