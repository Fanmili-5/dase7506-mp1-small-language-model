$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage87-stage71-direction-screen.json"
$Start = "runs/stage67-output-bias-s17/average.pt"
$End = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Output) { throw "Refusing to overwrite Stage87" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage87_stage71_direction.py `
    --start $Start --end $End --counts $Counts --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage87 direction screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage87 complete: diagnostic only; no export and no test scoring."
