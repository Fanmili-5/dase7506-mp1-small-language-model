$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Base = Join-Path $CodeRoot "runs\stage105-gated-singlepass\stage105-gated-singlepass.pt"
$Preflight = Join-Path $CodeRoot "results\stage207-preflight-a.json"
$Run = Join-Path $CodeRoot "runs\stage207-prefix-residual-pilot-s207017-a"
$LogDir = Join-Path $CodeRoot "job-logs\stage207-pilot-20260928-a"
if ((Test-Path $Run) -or (Test-Path $LogDir)) { throw "Refusing to overwrite Stage207 pilot" }
if (-not (Test-Path $Preflight)) { throw "Missing Stage207 preflight" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage207_prefix_residual_pilot"; status = $Status;
       error = $ErrorMessage; started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python -B -m unittest tests.test_stage207_prefix_residual -v
    if ($LASTEXITCODE -ne 0) { throw "Stage207 causal structure tests failed" }
    & $Python -B scripts\train_stage207_prefix_pilot.py `
        --base $Base --preflight $Preflight --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage207 fixed pilot failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
