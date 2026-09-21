param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("smoke", "stage1", "stage2", "stage3", "stage4", "stage4b", "stage5", "stage5b", "stage5c", "stage6", "stage7", "stage8", "stage9", "stage10")]
    [string]$Job,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunId
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
$TaskName = "MP1-$RunId"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task already exists; choose a new RunId: $TaskName"
}
if (Test-Path (Join-Path $CodeRoot "job-logs\$RunId")) {
    throw "Job logs already exist; choose a new RunId."
}
if (-not (Test-Path (Join-Path $CodeRoot ".venv\Scripts\python.exe"))) {
    throw "Install and validate the project environment before launching jobs."
}
$Runner = Join-Path $PSScriptRoot "run_windows_job.ps1"
$Arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Runner`" -Job $Job -RunId $RunId"
$Action = New-ScheduledTaskAction `
    -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument $Arguments -WorkingDirectory $CodeRoot
$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 6)
# One-off, on-demand job; no recurring trigger and no stored account password.
# Default battery safeguards are kept. Windows user must remain logged in.
Register-ScheduledTask -TaskName $TaskName -Action $Action -Principal $Principal `
    -Settings $Settings -Description "MP1 $Job; logs: $CodeRoot\job-logs\$RunId" | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output "Started task: $TaskName"
Write-Output "Status: $CodeRoot\job-logs\$RunId\status.json"
