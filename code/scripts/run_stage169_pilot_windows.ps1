$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Run = Join-Path $CodeRoot "runs\stage169-strong-student-pilot-s169017"
$LogDir = Join-Path $CodeRoot "job-logs\stage169-strong-student-pilot-20260925-a"
if (Test-Path $Run) { throw "Refusing to overwrite Stage169 pilot run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage169 pilot logs" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage169_pilot"; status = $Status; error = $ErrorMessage;
       started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python -m unittest tests.test_stage100_gate_features -v
    if ($LASTEXITCODE -ne 0) { throw "Gated-model tests failed" }
    & $Python scripts/train_stage169_strong_student_pilot.py `
        --stage105 runs/stage105-gated-singlepass/stage105-gated-singlepass.pt `
        --stage155 runs/stage155-neural-budget-full-s17/last-five-average.pt `
        --preflight runs/stage169-strong-student-preflight-v1.json --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage169 pilot run failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
