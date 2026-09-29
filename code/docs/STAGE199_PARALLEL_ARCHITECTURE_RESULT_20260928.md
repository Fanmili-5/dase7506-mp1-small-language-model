# Stage199 result: full-width parallel global/local mixer stopped at quality gate

Stage199 was a prospective reassessment of Stage177 under the course's
complete-predictor 5x CPU rule, rather than Stage177's earlier, stricter
1.25x feature-only screen. The plan and fixed endpoint audit were committed
before the new validation result. This reassessment did **not** claim a
resource pass: its 4.67x figure was only a timing projection.

The synthetic RTX 3070 Ti preflight completed two BF16 AdamW updates at
physical batch 32, with finite loss/gradients, maximum probability-mass
error `4.77e-7`, no observed future-prefix effect, and peak reserved VRAM
6.799 GB of 8.589 GB. No data split was opened. The structural Stage177
causality, zero-start and gradient unit tests also passed.

The fixed seed-17 pilot then completed 2,400 updates and **19,660,800**
primary train-target presentations in 785.70 training seconds. Its endpoint
was **1.514341121 BPB** on the complete 376,599-target, 1,148,007-byte
validation set. The same-seed, same-target, same-learning-rate Stage54
control was **1.519950369 BPB**, yielding **0.005609248 BPB** gain. This is
far below the predeclared **0.030 BPB** full-run admission gate. The
precommitted audit checked all scheduled 300-step validation points,
training-target count, source/data hashes, final checkpoint hash and the
absence of test scoring. Its result is
`rejected_below_predeclared_quality_gate`.

Consequently the Stage199 architecture is stopped: no 7,200-step training,
train-only count rebuild, compact export, full CPU/RAM/asset qualification,
method freeze or test score. The pilot checkpoint stays on the Windows
experiment host and is not an inference bundle. The protected Stage143
checkpoint and graph remain unchanged. This rejects the measured full-width
parallel architecture under this fixed recipe, not all global/local mixers.

Evidence: `code/results/stage199-evidence/preflight.json`, `run.json`,
`progress.json`, `metrics.json`, and `audit.json`.
