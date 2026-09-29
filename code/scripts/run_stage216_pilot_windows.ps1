$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Run = Join-Path $CodeRoot "runs\stage216-relative-bias-pilot-s17-a"
$LogDir = Join-Path $CodeRoot "job-logs\stage216-pilot-20260928-a"
$Preflight = Join-Path $CodeRoot "runs\stage216-preflight-a\result.json"
if ((Test-Path $Run) -or (Test-Path $LogDir)) { throw "Refusing to overwrite Stage216 pilot" }
if (-not (Test-Path $Preflight)) { throw "Missing Stage216 CPU/GPU preflight" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage216_relative_bias_pilot"; status = $Status;
       error = $ErrorMessage; started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python -B -m unittest tests.test_stage216_relative_bias -v
    if ($LASTEXITCODE -ne 0) { throw "Stage216 structural tests failed" }
    & $Python -B scripts/run_stage216_relative_bias_pilot.py --preflight $Preflight --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage216 fixed pilot failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
