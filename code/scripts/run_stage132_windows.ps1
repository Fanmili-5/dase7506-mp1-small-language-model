$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage132-local-successor-transfer"
$Checkpoint = "runs/stage115-order5-gate-export/stage115-order5-gate.pt"
$CheckpointSha = "902e4b21c9ddf3afda2258ae032fe852517716cccfd5341567b2fabc21910e76"
$ScannerSha = "3b0c9b575a5d27797e41cd9f659622c50033710f45c3e51fe7b3ad5ab28a7595"
if (Test-Path $Run) { throw "Refusing to overwrite Stage132" }
if ((Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLower() -ne $CheckpointSha) {
    throw "Stage115 checkpoint hash mismatch"
}
if ((Get-FileHash "scripts/scan_stage43_exact_local_cache.py" -Algorithm SHA256).Hash.ToLower() -ne $ScannerSha) {
    throw "Exact-cache scanner source hash mismatch"
}
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_exact_local_cache -v
if ($LASTEXITCODE -ne 0) { throw "Exact-cache tests failed" }
New-Item -ItemType Directory -Path $Run | Out-Null
& $Python scripts/scan_stage43_exact_local_cache.py --checkpoint $Checkpoint `
    --output "$Run/scan.json"
if ($LASTEXITCODE -ne 0) { throw "Stage132 scan failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage132 validation-only local-cache transfer screen complete."
