$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Run = Join-Path $CodeRoot "runs\stage204-hashed-bigram-pilot-s17-a"
$LogDir = Join-Path $CodeRoot "job-logs\stage204-pilot-20260928-a"
$Preflight = Join-Path $CodeRoot "runs\stage204-preflight-20260928-b\result.json"
if ((Test-Path $Run) -or (Test-Path $LogDir)) { throw "Refusing to overwrite Stage204 pilot" }
if (-not (Test-Path $Preflight)) { throw "Missing passed Stage204 preflight-b" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage204_hashed_bigram_pilot"; status = $Status; error = $ErrorMessage;
       started_utc = $StartedUtc; updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python -B -m unittest tests.test_stage204_hashed_bigram -v
    if ($LASTEXITCODE -ne 0) { throw "Stage204 structure tests failed" }
    & $Python -B scripts/run_stage204_hashed_bigram_pilot.py --preflight $Preflight --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage204 fixed pilot failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
