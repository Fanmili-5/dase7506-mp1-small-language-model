# Stage125: train the deployable neural/count mixture directly

The over-budget Stage91 ensemble scores 1.381623 BPB with fixed
Stage71/Stage76/order-five-MKN weights 0.5375/0.4250/0.0375. Stage92 distills
only its neural-only 0.55/0.45 part into one model, then adds count predictions
at inference; its best static order-five mixture remains at 1.401708 BPB.
Stage109's teacher-only continuation and Stage120's stronger *neural-only*
hard-label continuation both regress. A distinct explanation is that training
the neural distribution alone does not optimize the final deployable mixture.

Stage125 starts from the exact Stage92 average. The frozen teacher target is
the full Stage91 three-expert distribution. The student prediction is its
Stage94-calibrated neural distribution mixed with the same frozen train-only
order-five MKN expert at weight 0.0625. On supplied **training prefixes only**,
train with 0.75 teacher cross-entropy and 0.25 true next-token NLL of that
final student mixture. No teacher or count parameter updates. This replaces
the objective, not the inference graph or the seed of an existing recipe.

Before training, run a source-pinned single-batch numerical/throughput probe:
require finite loss and gradients, normalized teacher/student probabilities,
and GPU memory below 7 GiB. If safe, run one fixed seed-125017 trajectory of
1,500 steps, batch 24, AdamW peak LR 3e-6, weight decay 0.1, with validation
at 0/300/600/900/1200/1500 and fixed average of 900/1200/1500. The order-five
count expert and Stage94 calibration remain fixed for monitoring. Stop without
promotion if neither endpoint nor fixed average improves Stage92 on full
validation. Any gain still needs exact fused export and the three-repeat
5x CPU / 4 GiB RAM / 64 MiB asset audit. Stage85 stays the qualified fallback.
No validation or test label is used in gradients, and no test score is run.

## Numerical preflight correction

The first source-pinned probe had finite gradients and used only 2.41 GB GPU
memory, but BF16 arithmetic left the student final-mixture row sums off by
up to 0.001816 in log space, above the predeclared 0.001 threshold. It was
recorded as a failed preflight, not waved through. A second probe explicitly
subtracts the student final mixture's row-wise `logsumexp` *inside the
training loss only*. This restores a valid teacher/student cross-entropy
without changing the fixed CPU inference graph or scoring pipeline. The
first raw record remains in `../results/stage125-evidence/probe-original.json`;
the second uses a distinct output path and source hash.

The corrected probe passed: teacher/student maximum log-normalization errors
were 8.16e-7/2.98e-7, gradient norm 0.2093, peak GPU allocation 2.414 GB,
and finite combined loss 2.8914. Count construction took 0.207 s and the
GPU forward/backward 1.459 s for the fixed 24x256 batch. Its raw record is
`../results/stage125-evidence/probe-normalized.json`. The 1,500-step training
run is therefore allowed under the original feasibility rule.

## Completed training result

All 1,500 steps and 9,216,000 new training targets completed in 603.50 GPU
training seconds. The unchanged start scored **1.401707689** validation BPB.
The fixed monitoring points at 300/600/900/1200/1500 scored
1.401831935/1.401820078/1.401844660/1.401822694/1.401806218 BPB.
The prespecified three-checkpoint average (SHA-256
`4fdb334d40f922efd55eea0ae35e7a9f968cc2e1fc47b40fd7c8c06c71e1119b`)
scored **1.401822708**. Best remains step 0, so final-mixture-aware
distillation is rejected without export or a CPU resource audit. This
particular objective did not close the teacher-student gap. Raw metadata,
training history, and average score are under `../results/stage125-evidence/`.
No test score was run.
