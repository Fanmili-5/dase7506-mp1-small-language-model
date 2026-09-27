$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Run = Join-Path $CodeRoot "runs\stage202-ordered-pilot-s17-a"
$LogDir = Join-Path $CodeRoot "job-logs\stage202-ordered-20260928-a"
$Preflight = Join-Path $CodeRoot "job-logs\stage202-preflight-20260928.json"
if (Test-Path $Run) { throw "Refusing to overwrite Stage202 run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage202 logs" }
if (-not (Test-Path $Preflight)) { throw "Missing Stage202 GPU preflight" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage202_ordered_pilot"; status = $Status; error = $ErrorMessage;
       started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python -m unittest tests.test_stage202_ordered_hybrid -v
    if ($LASTEXITCODE -ne 0) { throw "Stage202 structure tests failed" }
    & $Python scripts/run_stage202_ordered_hybrid_pilot.py --preflight $Preflight --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage202 pilot failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
