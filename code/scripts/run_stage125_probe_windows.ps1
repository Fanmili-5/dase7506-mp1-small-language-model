$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage125-mixture-objective-normalized-probe.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage125 probe" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/probe_stage125_mixture_objective.py `
    --student "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --primary "runs/stage71-mixture-aware-s71017-d/average.pt" `
    --alternate "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt" `
    --counts "runs/stage25-kneser-ney/counts-min2/checkpoint.pt" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage125 probe failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
