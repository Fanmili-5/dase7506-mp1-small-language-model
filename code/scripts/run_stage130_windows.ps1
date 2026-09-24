$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Output = "runs/stage130-distillation-transfer-audit.json"
if (Test-Path $Output) { throw "Refusing to overwrite Stage130" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/audit_stage130_distillation_transfer.py `
    --student92 "runs/stage92-heterogeneous-distillation-s92017/average.pt" `
    --student129 "runs/stage129-merged-backbone-lora-s129017/average.pt" `
    --primary "runs/stage71-mixture-aware-s71017-d/average.pt" `
    --alternate "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt" `
    --counts "runs/stage25-kneser-ney/counts-min2/checkpoint.pt" `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Stage130 transfer audit failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage130 read-only audit complete; no test scoring."
