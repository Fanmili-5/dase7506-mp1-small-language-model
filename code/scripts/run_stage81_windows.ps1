$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage81-frequency-mixture-diagnostic.json"
$Neural = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
$Screen = "runs/stage79-stage71-calibration-refinement/screen.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage81" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/analyze_stage81_frequency_mixture.py `
    --neural $Neural --counts $Counts --screen $Screen --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage81 frequency-mixture diagnostic failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage81 complete: diagnostic only; no export and no test scoring."
