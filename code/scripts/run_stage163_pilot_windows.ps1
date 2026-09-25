$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Base = Join-Path $CodeRoot "runs\stage105-gated-singlepass\stage105-gated-singlepass.pt"
$Cache = Join-Path $CodeRoot "runs\stage146-stage143-target-logp-v1.npy"
$Run = Join-Path $CodeRoot "runs\stage163-compact-knn-v1"
$LogDir = Join-Path $CodeRoot "job-logs\stage163-compact-knn-20260925-a"
if (Test-Path $Run) { throw "Refusing to overwrite Stage163 run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage163 logs" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage163_compact_knn"; status = $Status; error = $ErrorMessage;
       started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python scripts/diagnose_stage163_compact_knn.py `
        --base $Base --stage143-cache $Cache --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage163 quality pilot failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
