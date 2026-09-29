$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = "$CodeRoot\.venv\Scripts\python.exe"
$Run = "runs/stage97-output-lora-fixed-teacher-s96017"
$Primary = "runs/stage71-mixture-aware-s71017-d/average.pt"
$Alternate = "runs/stage76-balanced-hybrid-continuation-s17/average-inference.pt"
$Counts = "runs/stage25-kneser-ney/counts-min2/checkpoint.pt"
if (Test-Path $Run) { throw "Refusing to overwrite Stage97" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Fixed files changed" }
& $Python scripts/train_stage96_output_lora_distillation.py `
    --primary $Primary --alternate $Alternate --counts $Counts --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw "Stage97 training failed" }
& $Python scripts/verify_fixed_files.py
if ($LASTEXITCODE -ne 0) { throw "Final fixed files changed" }
Write-Output "Stage97 complete: inspect corrected fixed-teacher output LoRA; no test scoring."
