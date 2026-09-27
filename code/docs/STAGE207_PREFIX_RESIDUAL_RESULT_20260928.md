# Stage207 result: explicit causal word-prefix residual fails

The [fixed pre-outcome plan](STAGE207_PREFIX_CONDITIONED_RESIDUAL_PLAN_20260928.md)
combined the observed ASCII current-word prefix with the frozen Transformer
hidden state and spelling-composed 2,048-token output rows. Unlike the
Stage191 count expert, it was context-conditioned; unlike Stage160, it
explicitly received the observed word prefix. No test text was opened by
the development loader, and no test score was produced.

## Feasibility screen only

The Windows synthetic/input-only preflight passed its gates: zero-start
maximum log-probability error `9.5367e-7`, normalization error `4.7684e-7`,
future-prefix and row-independence errors zero, conservative projected
assets `58,458,860` bytes, and eight-pair four-thread prefix-feature/head
median `0.096812` seconds per 32x256 batch. Three synthetic unit tests
passed on Mac and Windows. This does **not** establish complete predictor
CPU time, lifetime RAM, assets or trained quality.

## Fixed pilot endpoint

The Windows scheduled task exited 0. The run completed exactly 2,400 updates
at seed 207017, physical batch 16, with 9,830,400 primary train-target
presentations in 660.16 training seconds. Step zero reproduced the frozen
Stage143 full-validation control at approximately `1.399686` BPB. The
predeclared GPU FP32 step-2,400 endpoint was **1.4013225351170002 BPB** over all
376,599 validation targets and 1,148,007 raw bytes, a **regression of
0.001636373 BPB**. Both fixed validation halves regressed, with target-NLL
gains of `-511.983` and `-790.114` nats. The >=0.020 BPB quality gate fails.

The on-host residual checkpoint SHA-256 matched the run manifest:
`ba9abe92b9597098c78cfdbd8bb13f7251918c8785b7194ad9bc967bb78b8f1c`.
All 25 recorded source hashes match this checkout. Preflight, training
manifest, progress, full metrics and terminal job status are under
`../results/stage207-evidence/`; the full PowerShell transcript remains on
the Windows host because it contains the local Windows account name.

**Decision:** stop this exact lexical-prefix residual. Do not select an
intermediate checkpoint, vary character length, learning rate, dropout or
seed, export a trained CPU predictor, claim resource qualification, or score
the test split. The negative result does not prove that every hierarchical
word/character model fails; it does show that explicit word-prefix features
added to this frozen residual do not close the current quality gap. Stage143
remains the protected best qualified candidate: validation `1.399686162`,
already-frozen test `1.415657617`, above the student's 1.35/1.38 goals.
