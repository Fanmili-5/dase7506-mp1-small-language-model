$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Run = Join-Path $CodeRoot "runs\stage159-compact-complement-pilot-s17"
$Checkpoint = Join-Path $Run "checkpoint.pt"
$MixtureOutput = Join-Path $Run "fixed-mixture-validation.json"
$LogDir = Join-Path $CodeRoot "job-logs\stage159-compact-complement-pilot-20260925-a"
if (Test-Path $Run) { throw "Refusing to overwrite Stage159 pilot run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage159 pilot logs" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage159_pilot"; status = $Status; error = $ErrorMessage;
       started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python -m unittest tests.test_hybrid_conv_rdrop -v
    if ($LASTEXITCODE -ne 0) { throw "Hybrid R-Drop tests failed" }
    & $Python scripts/run_stage159_compact_complement_pilot.py --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage159 pilot run failed" }
    & $Python scripts/screen_stage159_compact_complement.py `
        --stage143-cache runs/stage146-stage143-target-logp-v1.npy `
        --compact $Checkpoint --output $MixtureOutput
    if ($LASTEXITCODE -ne 0) { throw "Stage159 fixed-mixture screen failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
