$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Stage 5B benchmarks the best-quality Stage-5A shape before any long run.
$BaselineCheckpoint = "runs/stage3-long-baseline-s17/checkpoint.pt"
$CandidateCheckpoint = "runs/stage5-w256_d6-s17/checkpoint.pt"
$ResourceOutput = "results/cpu-stage5-w256_d6.json"
foreach ($Path in @($BaselineCheckpoint, $CandidateCheckpoint)) {
    if (-not (Test-Path $Path)) { throw "Missing checkpoint: $Path" }
}
if (Test-Path $ResourceOutput) { throw "Resource output already exists: $ResourceOutput" }

& $Python scripts/benchmark_cpu.py `
    --baseline $BaselineCheckpoint --candidate $CandidateCheckpoint `
    --repeats 3 --threads 4 --output $ResourceOutput
if ($LASTEXITCODE -ne 0) { throw "Stage 5B CPU resource gate failed." }
Write-Output "Stage 5B complete. Validation only; no test evaluation."
