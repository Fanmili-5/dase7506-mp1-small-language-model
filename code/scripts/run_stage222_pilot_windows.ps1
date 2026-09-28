$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Preflight = Join-Path $CodeRoot "preflight\stage221-tied-depth-a\result.json"
$Frozen = Join-Path $CodeRoot "results\stage143-evidence\final.json"
$Run = Join-Path $CodeRoot "runs\stage222-course-aligned-tied-s17"
$LogDir = Join-Path $CodeRoot "job-logs\stage222-course-aligned-tied-20260928-a"
if (Test-Path $Run) { throw "Refusing to overwrite Stage222 run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage222 logs" }
if (-not (Test-Path $Preflight)) { throw "Stage221 preflight is missing" }
if (-not (Test-Path $Frozen)) { throw "Stage143 resource evidence is missing" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage222_course_aligned_tied_pilot"; status = $Status;
       error = $ErrorMessage; started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python -m unittest tests.test_stage221_tied_depth -v
    if ($LASTEXITCODE -ne 0) { throw "Stage221 structural tests failed" }
    & $Python scripts/run_stage222_course_aligned_tied_pilot.py `
        --preflight $Preflight --frozen-resource $Frozen --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage222 pilot failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
