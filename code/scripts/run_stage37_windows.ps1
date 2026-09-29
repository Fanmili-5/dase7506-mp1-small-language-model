$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage37-calibrated-qualified"
$Source = "runs/stage30-multi-token-mkn/best-mixture.pt"
$Baseline = "runs/stage3-long-baseline-s17/checkpoint-best.pt"
$ResumeQualifiedPreparation = Test-Path "$Run/equivalence.json"
if ((Test-Path $Run) -and -not $ResumeQualifiedPreparation) {
    throw "Stage37 directory exists without completed equivalence evidence"
}
if ($ResumeQualifiedPreparation) {
    $PreparedHash = (Get-FileHash "$Run/calibrated-collapsed.pt" -Algorithm SHA256).Hash.ToLower()
    if ($PreparedHash -ne "ed29352b1fca4c325beec7fcb30ab2b4a019a3f264b256945d119534c2380e43") {
        throw "Prepared Stage37 checkpoint changed"
    }
}
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_calibrated_collapsed -v
if ($LASTEXITCODE -ne 0) { throw "Calibrated collapsed tests failed" }
if (-not $ResumeQualifiedPreparation) {
    & $Python scripts/prepare_stage37_calibrated.py --source $Source --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage37 serialization/equivalence failed" }
}
& $Python evaluate.py --checkpoint "$Run/calibrated-collapsed.pt" --device cpu `
    --precision fp32 --threads 4 --split validation --output "$Run/validation-cpu-fp32.json"
if ($LASTEXITCODE -ne 0) { throw "Stage37 independent validation failed" }
& $Python scripts/benchmark_cpu.py --baseline $Baseline --candidate "$Run/calibrated-collapsed.pt" `
    --repeats 3 --threads 4 --output "$Run/resources.json"
if ($LASTEXITCODE -ne 0) { throw "Stage37 resource measurement failed" }
& $Python scripts/finalize_stage37.py --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage37 final audit failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage37 complete: inspect calibrated qualification; no test scoring."
