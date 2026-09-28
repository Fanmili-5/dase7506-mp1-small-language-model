$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Run = Join-Path $CodeRoot "runs\stage213-sliding-pilot-s17-a"
$LogDir = Join-Path $CodeRoot "job-logs\stage213-pilot-20260928-a"
$Preflight = Join-Path $CodeRoot "results\stage213-sliding-gpu-preflight-a.json"
if ((Test-Path $Run) -or (Test-Path $LogDir)) { throw "Refusing to overwrite Stage213 pilot" }
if (-not (Test-Path $Preflight)) { throw "Missing Stage213 GPU preflight" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage213_sliding_quality_pilot"; status = $Status;
       error = $ErrorMessage; started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python -B -m unittest tests.test_stage212_sliding_local tests.test_stage213_sliding_pilot -v
    if ($LASTEXITCODE -ne 0) { throw "Stage213 tests failed" }
    & $Python -B scripts/run_stage213_sliding_pilot.py --preflight $Preflight --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage213 fixed pilot failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
