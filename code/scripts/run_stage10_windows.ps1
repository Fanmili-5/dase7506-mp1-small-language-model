$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

# Formal pre-freeze resource benchmark for the Stage-9 selected checkpoint.
$BaselineCheckpoint = "runs/stage3-long-baseline-s17/checkpoint.pt"
$CandidateCheckpoint = "runs/stage8-long-drop010-w256_d6-s17/checkpoint-best.pt"
$ResourceOutput = "results/cpu-final-stage9-s17.json"
foreach ($Path in @($BaselineCheckpoint, $CandidateCheckpoint)) {
    if (-not (Test-Path $Path)) { throw "Missing checkpoint: $Path" }
}
if (Test-Path $ResourceOutput) { throw "Resource output already exists: $ResourceOutput" }

& $Python scripts/benchmark_cpu.py `
    --baseline $BaselineCheckpoint --candidate $CandidateCheckpoint `
    --repeats 3 --threads 4 --output $ResourceOutput
if ($LASTEXITCODE -ne 0) { throw "Final CPU resource gate failed." }
Write-Output "Stage 10 complete. Validation-only resource gate; no test evaluation."
