$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Checkpoint = Join-Path $CodeRoot "runs\stage143-openvino-singlecopy-v1\stage143-openvino-order6.pt"
$Output = Join-Path $CodeRoot "runs\stage203-class-context-20260928-a\result.json"
$LogDir = Join-Path $CodeRoot "job-logs\stage203-class-context-20260928-a"
if (Test-Path $Output) { throw "Refusing to overwrite Stage203 result" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage203 logs" }
if (-not (Test-Path $Checkpoint)) { throw "Missing pinned Stage143 checkpoint" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage203_class_context_validation"; status = $Status;
       error = $ErrorMessage; started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python -B -m unittest tests.test_stage203_class_context -v
    if ($LASTEXITCODE -ne 0) { throw "Stage203 structure tests failed" }
    & $Python -B scripts/diagnose_stage203_class_context.py --checkpoint $Checkpoint --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Stage203 diagnostic failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
