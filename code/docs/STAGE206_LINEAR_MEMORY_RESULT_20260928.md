# Stage206 result: causal linear-memory core fails the fixed quality gate

The [predeclared plan](STAGE206_LINEAR_MEMORY_PLAN_20260928.md) replaced
softmax-attention blocks 3 and 7 in the Stage54 backbone with rank-16
within-window positive-feature linear attention. This was a distinct core
mechanism, not a new seed or output calibration. The Windows job completed
normally (scheduled-task exit 0), and the checkpoint's on-host SHA-256
matched its recorded digest:
`11dfaac737247ca8f655d2c92d2b330ee64235d2a0c75c19d351b1ca8d42549f`.
All 25 recorded training-source hashes match this repository checkout.

## Feasibility, not quality

The synthetic, input-only preflight passed: maximum OpenVINO/PyTorch feature
error `3.8147e-6`, normalized output error `4.7684e-7`, future-prefix
error zero, independent-row error `4.7684e-7`, conservative projected assets
`55,351,618` bytes (<64 MiB), eight-pair median feature-time ratio `1.0230`
(<1.25), and a finite BF16 batch-32 gradient step using `5,295,527,936`
peak allocated GPU bytes (<7 GiB). These do **not** qualify a trained full
predictor on official CPU/RAM timing.

## Fixed matched-target endpoint

At seed 17, physical/effective batch 32 and the identical first 2,400 rates
of Stage54's 7,200-step schedule, the candidate presented `19,660,800`
primary training targets in `631.35` training seconds. Its complete GPU
FP32 validation endpoint over **376,599 targets / 1,148,007 UTF-8 bytes**
was **1.535373247 BPB**. The same-schedule Stage54 control was
**1.519950369 BPB**, so the replacement **regressed by 0.015422878 BPB**.
It fails the prespecified improvement gate of at least `0.040` BPB.

The complete validation trajectory at steps 300 through 2,400 is in the
raw `metrics.json`; every intermediate value is diagnostic, not a selectable
checkpoint. The preflight, run manifest, progress, final metrics and task
status are under `../results/stage206-evidence/`. The full console transcript
remains on the Windows training host and is excluded from the repository
because it contains the local Windows account name; the repository records
the three passing structural tests and the terminal task status.

**Decision:** stop Stage206. Do not change rank/layers/seed to search around
this negative result, extend to 7,200 steps, export a trained inference graph,
claim CPU/RAM qualification, or score the test split. This result does not
rule out all state-space or recurrent models; it rejects only this exact
rank-16 two-block linear-memory replacement under the fixed comparison.
Stage143 remains the protected best qualified candidate at validation
`1.399686162` and its already frozen test record `1.415657617`, neither of
which meets the student's `1.35`/`1.38` goals.
