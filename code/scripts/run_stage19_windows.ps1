$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage19-complementarity"
$Neural = "runs/stage18-regularization-s17/H-embedding-drop/average-inference.pt"
$Counts = "runs/stage16-ngram-min3/checkpoint.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage19" }
if ((Get-FileHash $Baseline -Algorithm SHA256).Hash.ToLower() -ne "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d") { throw "Baseline changed" }
if ((Get-FileHash $Neural -Algorithm SHA256).Hash.ToLower() -ne "2be8f4f4ac195593835aaa74f042b0b5d0f5473a46eb6d09fba99b3c11d263f7") { throw "H changed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed-file check failed" }
& $Python scripts/profile_structured_cpu.py --checkpoint $Neural --output "$Run/profile.json"
if ($LASTEXITCODE -ne 0) { throw "Profile failed" }
& $Python scripts/screen_stage19_mixture.py --neural $Neural --counts $Counts --run-dir "$Run/mixture"
if ($LASTEXITCODE -ne 0) { throw "Mixture failed" }
& $Python scripts/benchmark_cpu.py --baseline $Baseline --candidate $Neural --repeats 3 --output "$Run/h-resources.json"
if ($LASTEXITCODE -ne 0) { throw "H resource measurement failed" }
if (Test-Path "$Run/mixture/best-mixture.pt") {
    & $Python scripts/benchmark_cpu.py --baseline $Baseline --candidate "$Run/mixture/best-mixture.pt" --repeats 3 --output "$Run/mixture-resources.json"
    if ($LASTEXITCODE -ne 0) { throw "Hybrid resource measurement failed" }
}
& $Python scripts/diagnose_generalization.py --checkpoint $Neural --output "$Run/h-diagnostic.json" --device cpu
if ($LASTEXITCODE -ne 0) { throw "H diagnostic failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed-file check failed" }
Write-Output "Stage19 diagnostics completed; no training or test scoring. Resource failures are recorded, not waived."
