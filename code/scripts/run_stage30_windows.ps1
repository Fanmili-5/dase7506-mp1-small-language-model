$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage30-multi-token-mkn"
$Neural = "runs/stage26-multi-token-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage30" }
if ((Get-FileHash $Neural -Algorithm SHA256).Hash.ToLower() -ne "7e1187c61684be0b8e913c63ce30e03c24c52388460fd997fcce00b4dbdaa629") { throw "Stage26 neural changed" }
if ((Get-FileHash $Counts -Algorithm SHA256).Hash.ToLower() -ne "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2") { throw "Stage25 counts changed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/screen_stage30_multi_token_mkn.py --neural $Neural --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage30 scan failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage30 complete: Stage26 plus fixed modified-KN validation scan; no test scoring."
