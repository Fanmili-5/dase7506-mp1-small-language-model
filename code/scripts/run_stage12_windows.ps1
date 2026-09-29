$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $CodeRoot
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment is missing." }

$RunDirectory = "runs/stage12-long7200-drop010-w256_d6-s17"
if (Test-Path $RunDirectory) { throw "Refusing to overwrite existing run: $RunDirectory" }

& $Python train_experiment.py `
    --implementation student --config configs/student_w256_d6_drop010.json `
    --device cuda --precision auto --seed 17 --steps 7200 `
    --micro-batch-size 32 --grad-accum 1 --schedule baseline `
    --eval-every 300 --save-every 200 --keep-eval-checkpoints `
    --run-dir $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Stage-12 training failed." }

$AverageSpecs = @(
    @{ Name = "average-last2.pt"; Steps = @(6900, 7200) },
    @{ Name = "average-last3.pt"; Steps = @(6600, 6900, 7200) },
    @{ Name = "average-last5.pt"; Steps = @(6000, 6300, 6600, 6900, 7200) }
)
foreach ($Spec in $AverageSpecs) {
    $Arguments = @("scripts/average_checkpoints.py")
    foreach ($Step in $Spec.Steps) {
        $Arguments += @("--checkpoint", "$RunDirectory/checkpoints/step-$('{0:d6}' -f $Step).pt")
    }
    $Output = "$RunDirectory/$($Spec.Name)"
    $Arguments += @("--output", $Output)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint averaging failed for $($Spec.Name)." }
}

$Candidates = @("checkpoint-best.pt", "checkpoint.pt", "average-last2.pt", "average-last3.pt", "average-last5.pt")
foreach ($Name in $Candidates) {
    & $Python evaluate.py --checkpoint "$RunDirectory/$Name" --device cpu --precision fp32 `
        --threads 4 --split validation --output "$RunDirectory/$($Name.Replace('.pt', '-validation-cpu-fp32.json'))"
    if ($LASTEXITCODE -ne 0) { throw "CPU validation failed for $Name." }
}
Write-Output "Stage 12 complete. Validation-only long schedule and same-trajectory averages; no test evaluation."
