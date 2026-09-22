$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Root = "runs/stage25-kneser-ney"
$Neural = "runs/stage22-deep-supervision-s17/average-inference.pt"
if (Test-Path $Root) { throw "Refusing to overwrite Stage25" }
if ((Get-FileHash $Neural -Algorithm SHA256).Hash.ToLower() -ne "05f20e68313a3c41c4a1116bc3d335ad688507c933fb60e9176a83d8953e5269") { throw "Stage22 neural checkpoint changed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python -m unittest tests.test_kneser_ney -v
if ($LASTEXITCODE -ne 0) { throw "Modified-KN tests failed" }
New-Item -ItemType Directory -Path $Root | Out-Null
foreach ($MinCount in @(2, 3)) {
    $CountDir = "$Root/counts-min$MinCount"
    $ScanDir = "$Root/scan-min$MinCount"
    & $Python scripts/build_kneser_ney.py --output-dir $CountDir --min-count $MinCount
    if ($LASTEXITCODE -ne 0) { throw "Modified-KN build failed for min-count $MinCount" }
    & $Python scripts/screen_stage25_kneser_ney.py --neural $Neural `
        --counts "$CountDir/checkpoint.pt" --run-dir $ScanDir
    if ($LASTEXITCODE -ne 0) { throw "Modified-KN screen failed for min-count $MinCount" }
}
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage25 complete: two modified-KN quality screens; no resource claim or test scoring."
