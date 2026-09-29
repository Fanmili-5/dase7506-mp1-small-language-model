param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("smoke", "stage1", "stage2", "stage3", "stage4", "stage4b", "stage5", "stage5b", "stage5c", "stage6", "stage7", "stage8", "stage9", "stage10", "stage11", "stage12", "stage14", "stage15", "stage17", "stage18", "stage19", "stage20", "stage21", "stage22", "stage23", "stage24", "stage25", "stage26", "stage27", "stage28", "stage30", "stage31", "stage32", "stage33", "stage34", "stage35", "stage36", "stage37", "stage38", "stage39", "stage40", "stage41", "stage42", "stage43", "stage44", "stage45", "stage46", "stage47", "stage48", "stage49", "stage50", "stage51", "stage52", "stage53", "stage54", "stage55", "stage56", "stage57", "stage58", "stage59", "stage60", "stage61", "stage62", "stage63", "stage64", "stage65", "stage66", "stage67", "stage68", "stage69", "stage70", "stage71", "stage72", "stage73", "stage74", "stage75", "stage76", "stage77", "stage78", "stage79", "stage80", "stage81", "stage82", "stage83", "stage84", "stage85", "stage86", "stage87", "stage88", "stage89", "stage90", "stage91", "stage92", "stage93", "stage94", "stage95", "stage96", "stage97", "stage98")]
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
    } elseif ($Job -eq "stage21") {
        & "$PSScriptRoot\run_stage21_windows.ps1"
    } elseif ($Job -eq "stage22") {
        & "$PSScriptRoot\run_stage22_windows.ps1"
    } elseif ($Job -eq "stage23") {
        & "$PSScriptRoot\run_stage23_windows.ps1"
    } elseif ($Job -eq "stage24") {
        & "$PSScriptRoot\run_stage24_windows.ps1"
    } elseif ($Job -eq "stage25") {
        & "$PSScriptRoot\run_stage25_windows.ps1"
    } elseif ($Job -eq "stage26") {
        & "$PSScriptRoot\run_stage26_windows.ps1"
    } elseif ($Job -eq "stage27") {
        & "$PSScriptRoot\run_stage27_windows.ps1"
    } elseif ($Job -eq "stage28") {
        & "$PSScriptRoot\run_stage28_windows.ps1"
    } elseif ($Job -eq "stage30") {
        & "$PSScriptRoot\run_stage30_windows.ps1"
    } elseif ($Job -eq "stage31") {
        & "$PSScriptRoot\run_stage31_windows.ps1"
    } elseif ($Job -eq "stage32") {
        & "$PSScriptRoot\run_stage32_windows.ps1"
    } elseif ($Job -eq "stage33") {
        & "$PSScriptRoot\run_stage33_windows.ps1"
    } elseif ($Job -eq "stage34") {
        & "$PSScriptRoot\run_stage34_windows.ps1"
    } elseif ($Job -eq "stage35") {
        & "$PSScriptRoot\run_stage35_windows.ps1"
    } elseif ($Job -eq "stage36") {
        & "$PSScriptRoot\run_stage36_windows.ps1"
    } elseif ($Job -eq "stage37") {
        & "$PSScriptRoot\run_stage37_windows.ps1"
    } elseif ($Job -eq "stage38") {
        & "$PSScriptRoot\run_stage38_windows.ps1"
    } elseif ($Job -eq "stage39") {
        & "$PSScriptRoot\run_stage39_windows.ps1"
    } elseif ($Job -eq "stage40") {
        & "$PSScriptRoot\run_stage40_windows.ps1"
    } elseif ($Job -eq "stage41") {
        & "$PSScriptRoot\run_stage41_windows.ps1"
    } elseif ($Job -eq "stage42") {
        & "$PSScriptRoot\run_stage42_windows.ps1"
    } elseif ($Job -eq "stage43") {
        & "$PSScriptRoot\run_stage43_windows.ps1"
    } elseif ($Job -eq "stage44") {
        & "$PSScriptRoot\run_stage44_windows.ps1"
    } elseif ($Job -eq "stage45") {
        & "$PSScriptRoot\run_stage45_windows.ps1"
    } elseif ($Job -eq "stage46") {
        & "$PSScriptRoot\run_stage46_windows.ps1"
    } elseif ($Job -eq "stage47") {
        & "$PSScriptRoot\run_stage47_windows.ps1"
    } elseif ($Job -eq "stage48") {
        & "$PSScriptRoot\run_stage48_windows.ps1"
    } elseif ($Job -eq "stage49") {
        & "$PSScriptRoot\run_stage49_windows.ps1"
    } elseif ($Job -eq "stage50") {
        & "$PSScriptRoot\run_stage50_windows.ps1"
    } elseif ($Job -eq "stage51") {
        & "$PSScriptRoot\run_stage51_windows.ps1"
    } elseif ($Job -eq "stage52") {
        & "$PSScriptRoot\run_stage52_windows.ps1"
    } elseif ($Job -eq "stage53") {
        & "$PSScriptRoot\run_stage53_windows.ps1"
    } elseif ($Job -eq "stage54") {
        & "$PSScriptRoot\run_stage54_windows.ps1"
    } elseif ($Job -eq "stage55") {
        & "$PSScriptRoot\run_stage55_windows.ps1"
    } elseif ($Job -eq "stage56") {
        & "$PSScriptRoot\run_stage56_windows.ps1"
    } elseif ($Job -eq "stage57") {
        & "$PSScriptRoot\run_stage57_windows.ps1"
    } elseif ($Job -eq "stage58") {
        & "$PSScriptRoot\run_stage58_windows.ps1"
    } elseif ($Job -eq "stage59") {
        & "$PSScriptRoot\run_stage59_windows.ps1"
    } elseif ($Job -eq "stage60") {
        & "$PSScriptRoot\run_stage60_windows.ps1"
    } elseif ($Job -eq "stage61") {
        & "$PSScriptRoot\run_stage61_windows.ps1"
    } elseif ($Job -eq "stage62") {
        & "$PSScriptRoot\run_stage62_windows.ps1"
    } elseif ($Job -eq "stage63") {
        & "$PSScriptRoot\run_stage63_windows.ps1"
    } elseif ($Job -eq "stage64") {
        & "$PSScriptRoot\run_stage64_windows.ps1"
    } elseif ($Job -eq "stage65") {
        & "$PSScriptRoot\run_stage65_windows.ps1"
    } elseif ($Job -eq "stage66") {
        & "$PSScriptRoot\run_stage66_windows.ps1"
    } elseif ($Job -eq "stage67") {
        & "$PSScriptRoot\run_stage67_windows.ps1"
    } elseif ($Job -eq "stage68") {
        & "$PSScriptRoot\run_stage68_windows.ps1"
    } elseif ($Job -eq "stage69") {
        & "$PSScriptRoot\run_stage69_windows.ps1"
    } elseif ($Job -eq "stage70") {
        & "$PSScriptRoot\run_stage70_windows.ps1"
    } elseif ($Job -eq "stage71") {
        & "$PSScriptRoot\run_stage71_windows.ps1"
    } elseif ($Job -eq "stage72") {
        & "$PSScriptRoot\run_stage72_windows.ps1"
    } elseif ($Job -eq "stage73") {
        & "$PSScriptRoot\run_stage73_windows.ps1"
    } elseif ($Job -eq "stage74") {
        & "$PSScriptRoot\run_stage74_windows.ps1"
    } elseif ($Job -eq "stage75") {
        & "$PSScriptRoot\run_stage75_windows.ps1"
    } elseif ($Job -eq "stage76") {
        & "$PSScriptRoot\run_stage76_windows.ps1"
    } elseif ($Job -eq "stage77") {
        & "$PSScriptRoot\run_stage77_windows.ps1"
    } elseif ($Job -eq "stage78") {
        & "$PSScriptRoot\run_stage78_windows.ps1"
    } elseif ($Job -eq "stage79") {
        & "$PSScriptRoot\run_stage79_windows.ps1"
    } elseif ($Job -eq "stage80") {
        & "$PSScriptRoot\run_stage80_windows.ps1"
    } elseif ($Job -eq "stage81") {
        & "$PSScriptRoot\run_stage81_windows.ps1"
    } elseif ($Job -eq "stage82") {
        & "$PSScriptRoot\run_stage82_windows.ps1"
    } elseif ($Job -eq "stage83") {
        & "$PSScriptRoot\run_stage83_windows.ps1"
    } elseif ($Job -eq "stage84") {
        & "$PSScriptRoot\run_stage84_windows.ps1"
    } elseif ($Job -eq "stage85") {
        & "$PSScriptRoot\run_stage85_windows.ps1"
    } elseif ($Job -eq "stage86") {
        & "$PSScriptRoot\run_stage86_windows.ps1"
    } elseif ($Job -eq "stage87") {
        & "$PSScriptRoot\run_stage87_windows.ps1"
    } elseif ($Job -eq "stage88") {
        & "$PSScriptRoot\run_stage88_windows.ps1"
    } elseif ($Job -eq "stage89") {
        & "$PSScriptRoot\run_stage89_windows.ps1"
    } elseif ($Job -eq "stage90") {
        & "$PSScriptRoot\run_stage90_windows.ps1"
    } elseif ($Job -eq "stage91") {
        & "$PSScriptRoot\run_stage91_windows.ps1"
    } elseif ($Job -eq "stage92") {
        & "$PSScriptRoot\run_stage92_windows.ps1"
    } elseif ($Job -eq "stage93") {
        & "$PSScriptRoot\run_stage93_windows.ps1"
    } elseif ($Job -eq "stage94") {
        & "$PSScriptRoot\run_stage94_windows.ps1"
    } elseif ($Job -eq "stage95") {
        & "$PSScriptRoot\run_stage95_windows.ps1"
    } elseif ($Job -eq "stage96") {
        & "$PSScriptRoot\run_stage96_windows.ps1"
    } elseif ($Job -eq "stage97") {
        & "$PSScriptRoot\run_stage97_windows.ps1"
    } else {
        & "$PSScriptRoot\run_stage98_windows.ps1"
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
