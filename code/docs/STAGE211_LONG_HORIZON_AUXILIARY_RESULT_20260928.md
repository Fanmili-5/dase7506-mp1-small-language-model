# Stage211: long-horizon auxiliary pilot — stopped

The predeclared Stage211 experiment in
[`STAGE211_LONG_HORIZON_AUXILIARY_PLAN_20260928.md`](STAGE211_LONG_HORIZON_AUXILIARY_PLAN_20260928.md)
completed on the Windows RTX 3070 Ti on 28 September 2026. It changed the
Stage54 train-only future-token offsets from `[2, 3]` to
`[2, 3, 8, 16, 32, 64]`, keeping the total auxiliary loss weight at `0.2`.
The neural inference architecture and all shared seed-17 starting weights
were unchanged. The course tokenizer, train/validation splits and complete
validation scorer were unchanged. No new test evaluation was run.

The synthetic preflight passed: step-zero inference and exported inference
state were exactly equal to the control; normalization, causality and
independent-row checks passed. One BF16 batch-32 update had finite loss and
gradients with 6,311,428,096 bytes peak allocated GPU memory. This was a
feasibility check, not evidence of better language modeling.

The fixed 2,400-step pilot processed 19,660,800 primary next-token targets.
Its source hashes matched all 24 local source files after collection. The
sampler used the same seed-17 draw sequence as Stage54 and the same first
2,400 steps of its 7,200-step learning-rate schedule. Two sampled starts
were clamped to accommodate the longer future-label span; therefore exactly
matched primary contexts are not claimed for those two of 76,800 windows.

| Complete GPU FP32 validation | 2,400-step BPB |
| --- | ---: |
| Stage54 same-target control | 1.5199503686120217 |
| Stage211 long-horizon auxiliary | 1.5203222202557127 |

Stage211 **worsened** BPB by 0.000371852. It misses the predeclared
`>=0.030` improvement gate by a wide margin, so there is no 7,200-step
continuation, inference export, full CPU/RAM/asset qualification or new test
score. The protected Stage143 candidate remains the best resource-qualified
model at 1.399686162 validation BPB and 1.415657617 frozen test BPB;
neither meets the student's `<1.35` target or `<1.38` minimum test score.

Primary evidence: [preflight](../results/stage211-preflight-a.json),
[run plan](../results/stage211-pilot-run.json),
[metrics](../results/stage211-pilot-metrics.json),
[progress](../results/stage211-pilot-progress.json), and
[job status](../results/stage211-pilot-job-status.json). The candidate
checkpoint remains in the Windows pilot run directory and was not promoted.
