param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("smoke", "stage1", "stage2", "stage3", "stage4", "stage4b", "stage5", "stage5b", "stage5c", "stage6", "stage7", "stage8", "stage9", "stage10", "stage11", "stage12", "stage14", "stage15", "stage17", "stage18", "stage19", "stage20", "stage21")]
    [string]$Job,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunId
)

# Intended for a one-off Windows Scheduled Task using the existing interactive
# user session. No password is stored, and the job does not depend on SSH staying
# connected. It does not automatically resume after a Windows reboot/logoff.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUNBUFFERED = "1"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$LogDirectory = Join-Path $CodeRoot "job-logs\$RunId"
if (Test-Path $LogDirectory) { throw "Refusing to overwrite job logs: $LogDirectory" }
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$StartedUtc = (Get-Date).ToUniversalTime().ToString("o")
$StatusFile = Join-Path $LogDirectory "status.json"
function Write-JobStatus([string]$Status, [int]$ExitCode, [string]$ErrorMessage) {
    $State = @{
        job = $Job; run_id = $RunId; status = $Status; exit_code = $ExitCode
        started_utc = $StartedUtc; updated_utc = (Get-Date).ToUniversalTime().ToString("o")
        error = $ErrorMessage
    }
    $State | ConvertTo-Json | Set-Content -LiteralPath "$StatusFile.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$StatusFile.tmp" -Destination $StatusFile -Force
}

$ResultCode = 1
Start-Transcript -Path (Join-Path $LogDirectory "console.log")
try {
    Write-JobStatus "running" 0 ""
    if ($Job -eq "smoke") {
        & "$PSScriptRoot\smoke_windows_cuda.ps1"
    } elseif ($Job -eq "stage1") {
        & "$PSScriptRoot\run_stage1_windows.ps1"
    } elseif ($Job -eq "stage2") {
        & "$PSScriptRoot\run_stage2_windows.ps1"
    } elseif ($Job -eq "stage3") {
        & "$PSScriptRoot\run_stage3_windows.ps1"
    } elseif ($Job -eq "stage4") {
        & "$PSScriptRoot\run_stage4_windows.ps1"
    } elseif ($Job -eq "stage4b") {
        & "$PSScriptRoot\run_stage4b_windows.ps1"
    } elseif ($Job -eq "stage5") {
        & "$PSScriptRoot\run_stage5_windows.ps1"
    } elseif ($Job -eq "stage5b") {
        & "$PSScriptRoot\run_stage5b_windows.ps1"
    } elseif ($Job -eq "stage5c") {
        & "$PSScriptRoot\run_stage5c_windows.ps1"
    } elseif ($Job -eq "stage6") {
        & "$PSScriptRoot\run_stage6_windows.ps1"
    } elseif ($Job -eq "stage7") {
        & "$PSScriptRoot\run_stage7_windows.ps1"
    } elseif ($Job -eq "stage8") {
        & "$PSScriptRoot\run_stage8_windows.ps1"
    } elseif ($Job -eq "stage9") {
        & "$PSScriptRoot\run_stage9_windows.ps1"
    } elseif ($Job -eq "stage10") {
        & "$PSScriptRoot\run_stage10_windows.ps1"
    } elseif ($Job -eq "stage11") {
        & "$PSScriptRoot\run_stage11_windows.ps1"
    } elseif ($Job -eq "stage12") {
        & "$PSScriptRoot\run_stage12_windows.ps1"
    } elseif ($Job -eq "stage14") {
        & "$PSScriptRoot\run_stage14_windows.ps1"
    } elseif ($Job -eq "stage15") {
        & "$PSScriptRoot\run_stage15_windows.ps1"
    } elseif ($Job -eq "stage17") {
        & "$PSScriptRoot\run_stage17_windows.ps1"
    } elseif ($Job -eq "stage18") {
        & "$PSScriptRoot\run_stage18_windows.ps1"
    } elseif ($Job -eq "stage19") {
        & "$PSScriptRoot\run_stage19_windows.ps1"
    } elseif ($Job -eq "stage20") {
        & "$PSScriptRoot\run_stage20_windows.ps1"
    } else {
        & "$PSScriptRoot\run_stage21_windows.ps1"
    }
    Write-JobStatus "completed" 0 ""
    $ResultCode = 0
} catch {
    $Failure = $_ | Out-String
    Write-Output $Failure
    Write-JobStatus "failed" 1 $Failure
} finally {
    Stop-Transcript
}
exit $ResultCode
