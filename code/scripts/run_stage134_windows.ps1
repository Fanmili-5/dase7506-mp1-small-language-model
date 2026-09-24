$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage134-mkldnn-ffn-pilot.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage134" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/probe_stage134_mkldnn_ffn.py `
    --neural "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage134 pilot failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage134 input-only oneDNN FFN pilot complete."
