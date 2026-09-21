param(
    [string]$InstallerDirectory = "$env:USERPROFILE\mp1-installers"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (Test-Path $Python312) {
    & $Python312 --version
    if ($LASTEXITCODE -ne 0) { throw "Existing Python 3.12 executable failed." }
    exit 0
}

# Last python.org Windows installer for the assignment's Python 3.12 series.
# Install alongside existing Python; do not change PATH or file associations.
New-Item -ItemType Directory -Force -Path $InstallerDirectory | Out-Null
$Installer = Join-Path $InstallerDirectory "python-3.12.10-amd64.exe"
Invoke-WebRequest -UseBasicParsing `
    -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" `
    -OutFile $Installer
$Signature = Get-AuthenticodeSignature -FilePath $Installer
if ($Signature.Status -ne "Valid" -or $Signature.SignerCertificate.Subject -notmatch "Python Software Foundation") {
    throw "Python installer signature validation failed. Do not run this installer."
}
Get-FileHash -Algorithm SHA256 -LiteralPath $Installer | Format-List
$Process = Start-Process -FilePath $Installer -Wait -PassThru -ArgumentList @(
    "/quiet", "InstallAllUsers=0", "PrependPath=0", "AssociateFiles=0",
    "Include_launcher=0", "Include_test=0", "Include_doc=0", "Include_pip=1"
)
if ($Process.ExitCode -notin @(0, 3010)) { throw "Python installer failed: $($Process.ExitCode)" }
if (-not (Test-Path $Python312)) { throw "Installed Python was not found at expected path." }
& $Python312 --version
if ($LASTEXITCODE -ne 0) { throw "Installed Python is not usable." }
py -0p
if ($LASTEXITCODE -ne 0) { throw "Python Launcher verification failed." }
