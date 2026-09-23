$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage50-confidence-gate"
$Neural = "runs/stage47-rdrop-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage50" }
if (-not (Test-Path $Neural)) { throw "Stage47 exported average is required" }
if ((Get-FileHash $Counts -Algorithm SHA256).Hash.ToLower() -ne "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2") {
    throw "Stage25 count expert changed"
}
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
New-Item -ItemType Directory -Path $Run | Out-Null
& $Python scripts/analyze_stage50_confidence_gate.py `
    --neural $Neural --counts $Counts --output "$Run/diagnostic.json"
if ($LASTEXITCODE -ne 0) { throw "Stage50 diagnostic failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage50 complete: validation-only cross-fit gate diagnostic; no export or test scoring."
