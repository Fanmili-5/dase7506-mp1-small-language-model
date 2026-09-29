$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Neural = "runs/stage92-heterogeneous-distillation-s92017/average.pt"
$Counts = "runs/stage73-order6-b/counts/checkpoint.pt"
$Output = "runs/stage119-ffn-pruning-screen.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage119" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage119_ffn_pruning.py `
    --neural $Neural --counts $Counts --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage119 pruning screen failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage119 done: inspect before any pruning repair; no test scoring."
