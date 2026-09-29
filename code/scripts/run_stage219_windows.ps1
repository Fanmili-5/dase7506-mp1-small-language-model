$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Start = Join-Path $CodeRoot "runs\stage155-neural-budget-full-s17\last-five-average.pt"
$Run = Join-Path $CodeRoot "runs\stage219-stage155-continuation-s17"
$LogDir = Join-Path $CodeRoot "job-logs\stage219-stage155-continuation-20260928-a"
if (Test-Path $Run) { throw "Refusing to overwrite Stage219 run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage219 logs" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage219_stage155_continuation"; status = $Status;
       error = $ErrorMessage; started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python scripts/train_stage219_stage155_continuation.py --start $Start --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage219 continuation failed" }
    $Average = Join-Path $Run "last-five-average.pt"
    & $Python scripts/average_checkpoints.py `
        --checkpoint "$Run/checkpoints/step-003600.pt" `
        --checkpoint "$Run/checkpoints/step-003900.pt" `
        --checkpoint "$Run/checkpoints/step-004200.pt" `
        --checkpoint "$Run/checkpoints/step-004500.pt" `
        --checkpoint "$Run/checkpoints/step-004800.pt" `
        --output $Average
    if ($LASTEXITCODE -ne 0) { throw "Stage219 averaging failed" }
    & $Python scripts/score_stage219_average.py `
        --checkpoint $Average --output "$Run/last-five-average-validation-gpu-fp32.json"
    if ($LASTEXITCODE -ne 0) { throw "Stage219 average validation failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
