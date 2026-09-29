$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Run = Join-Path $CodeRoot "runs\stage179-dropout20-s17"
$LogDir = Join-Path $CodeRoot "job-logs\stage179-dropout20-20260926-a"
if (Test-Path $Run) { throw "Refusing to overwrite Stage179 run" }
if (Test-Path $LogDir) { throw "Refusing to overwrite Stage179 logs" }
New-Item -ItemType Directory -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "status.json"
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
function Write-Status([string]$Status, [string]$ErrorMessage) {
    @{ job = "stage179_dropout20"; status = $Status; error = $ErrorMessage;
       started_utc = $StartedUtc;
       updated_utc = (Get-Date).ToUniversalTime().ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusFile -Encoding UTF8
}
Start-Transcript -Path (Join-Path $LogDir "console.log") | Out-Null
try {
    Write-Status "running" ""
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Fixed-file verification failed" }
    & $Python -m unittest tests.test_hybrid_conv_rdrop tests.test_stage179_dropout20 -v
    if ($LASTEXITCODE -ne 0) { throw "Stage179 structural tests failed" }
    & $Python scripts/train_stage179_dropout20.py --run-dir $Run
    if ($LASTEXITCODE -ne 0) { throw "Stage179 training failed" }
    $Average = Join-Path $Run "average-training.pt"
    $Arguments = @("scripts/average_checkpoints.py")
    foreach ($Step in @(6000, 6300, 6600, 6900, 7200)) {
        $Arguments += @("--checkpoint", (Join-Path $Run ("checkpoints/step-{0:d6}.pt" -f $Step)))
    }
    $Arguments += @("--output", $Average)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage179 averaging failed" }
    $Exported = Join-Path $Run "average-inference.pt"
    & $Python scripts/export_stage54_hybrid_conv_rdrop.py --checkpoint $Average --output $Exported
    if ($LASTEXITCODE -ne 0) { throw "Stage179 export failed" }
    & $Python evaluate.py --checkpoint $Exported --device cpu --precision fp32 --threads 4 `
        --split validation --output (Join-Path $Run "average-validation-cpu-fp32.json")
    if ($LASTEXITCODE -ne 0) { throw "Stage179 CPU validation failed" }
    & $Python scripts/verify_fixed_files.py
    if ($LASTEXITCODE -ne 0) { throw "Final fixed-file verification failed" }
    Write-Status "completed" ""
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
