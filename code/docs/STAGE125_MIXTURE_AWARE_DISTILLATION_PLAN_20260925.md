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
