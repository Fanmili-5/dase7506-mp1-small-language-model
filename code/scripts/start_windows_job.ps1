param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("smoke", "stage1", "stage2", "stage3", "stage4", "stage4b", "stage5", "stage5b", "stage5c", "stage6", "stage7", "stage8", "stage9", "stage10", "stage11", "stage12", "stage14", "stage15", "stage17", "stage18", "stage19", "stage20", "stage21", "stage22", "stage23", "stage24", "stage25", "stage26", "stage27", "stage28", "stage30", "stage31", "stage32", "stage33", "stage34", "stage35", "stage36", "stage37", "stage38", "stage39", "stage40", "stage41", "stage42", "stage43", "stage44", "stage45", "stage46", "stage47", "stage48", "stage49", "stage50", "stage51", "stage52", "stage53", "stage54", "stage55", "stage56", "stage57", "stage58", "stage59", "stage60", "stage61", "stage62", "stage63", "stage64", "stage65", "stage66", "stage67", "stage68", "stage69", "stage70", "stage71", "stage72", "stage73", "stage74", "stage75", "stage76", "stage77", "stage78", "stage79", "stage80", "stage81", "stage82", "stage97", "stage98")]
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
