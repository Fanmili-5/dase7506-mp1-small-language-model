$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Preflight = Join-Path $CodeRoot "results\stage223-bigram-preflight.json"
$Run = Join-Path $CodeRoot "runs\stage223-bigram-input-s17"
$LogDir = Join-Path $CodeRoot "job-logs\stage223-bigram-input-20260928-a"
if (Test-Path $Run) { throw "Refusing to overwrite Stage223 run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage223 logs" }
if (-not (Test-Path $Preflight)) { throw "Stage223 preflight is missing" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage223_bigram_input_pilot"; status = $Status;
       error = $ErrorMessage; started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python -m unittest tests.test_stage223_bigram_input -v
    if ($LASTEXITCODE -ne 0) { throw "Stage223 structural tests failed" }
    & $Python scripts/run_stage223_bigram_pilot.py `
        --preflight $Preflight --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage223 pilot failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
