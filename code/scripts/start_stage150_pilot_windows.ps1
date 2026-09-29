$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
$TaskName = "MP1-stage150-depth10-pilot-20260925-a"
$Run = Join-Path $CodeRoot "runs\stage150-depth10-pilot-s17"
$Logs = Join-Path $CodeRoot "job-logs\stage150-depth10-pilot-20260925-a"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Stage150 pilot task already exists"
}
if ((Test-Path $Run) -or (Test-Path $Logs)) {
    throw "Stage150 pilot run or logs already exist"
}
$Runner = Join-Path $PSScriptRoot "run_stage150_pilot_windows.ps1"
$Arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Runner`""
$Action = New-ScheduledTaskAction `
    -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument $Arguments -WorkingDirectory $CodeRoot
$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Principal $Principal `
    -Settings $Settings -Description "Validation-only Stage150 depth-10 pilot" | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output "Started $TaskName; status: $Logs\status.json"
