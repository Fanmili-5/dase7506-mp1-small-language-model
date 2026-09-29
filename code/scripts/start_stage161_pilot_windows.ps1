$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
$TaskName = "MP1-stage161-frequency-focus-pilot-20260925-a"
$Run = Join-Path $CodeRoot "runs\stage161-frequency-focus-pilot-s17"
$Logs = Join-Path $CodeRoot "job-logs\stage161-frequency-focus-pilot-20260925-a"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Stage161 pilot task already exists"
}
if ((Test-Path $Run) -or (Test-Path $Logs)) {
    throw "Stage161 pilot run or logs already exist"
}
$Runner = Join-Path $PSScriptRoot "run_stage161_pilot_windows.ps1"
$Arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Runner`""
$Action = New-ScheduledTaskAction `
    -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument $Arguments -WorkingDirectory $CodeRoot
$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Principal $Principal `
    -Settings $Settings -Description "Matched Stage161 frequency-focus validation pilot" | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output "Started $TaskName; status: $Logs\status.json"
