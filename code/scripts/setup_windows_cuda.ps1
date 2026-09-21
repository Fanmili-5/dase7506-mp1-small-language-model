param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) not found. Install 64-bit Python 3.12 first."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py "-$PythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $Python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
& $Python -m pip install numpy==2.5.3 tokenizers==0.21.4
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

& $Python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Contract tests failed." }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed-file integrity check failed." }
& $Python scripts/check_environment.py --config configs/student_control.json --require-cuda
if ($LASTEXITCODE -ne 0) { throw "CUDA environment check failed." }

Write-Host "Windows CUDA environment is ready." -ForegroundColor Green
