# Stage157: transfer the complementary teacher into one deployable neural model

Stage156's fixed Stage143/Stage155 equal-probability diagnostic reached
1.3761826642 validation BPB, passing its preregistered value-of-information
gate. It cannot be deployed because the combined assets exceed 64 MiB.
Stage157 asks whether a **single Stage155-capacity neural student** can absorb
part of that complementary distribution using only supplied training prefixes.
The student's eventual compact FP32 OpenVINO graph/head would be the only
inference asset; neither teacher, count table nor validation cache would ship.

The Stage105 eager PyTorch checkpoint is an algebraically equivalent source
for Stage143's OpenVINO predictor (complete validation differed by about
1e-9 BPB). Freeze the Stage105 and Stage155-average checkpoints. Teacher
distribution on each sampled train prefix is their 50:50 mixture. Initialize
the separate student from the Stage155 average. Use full-distribution
teacher cross-entropy plus the actual train next-token NLL; no validation
target is a gradient input. The teacher is fixed as training proceeds.

Before any pilot, verify checkpoint hashes, compare Stage105 CUDA target
log-probabilities on the first 8 validation windows against the hash-pinned
Stage143 cache, check the teacher distribution normalizes, and run exactly
one Stage155-student update with effective/physical batch 16 and context 256
on the 8-GiB GPU. Require maximum Stage105-vs-Stage143 target-logp error
<=3e-4, normalization error <=1e-3, peak allocated CUDA memory <=6.5 GB,
and a one-step wall time <=3 seconds after initialization. If any fails,
redesign the teacher execution without starting a training run.

If feasible, one seed-157017 **900-update pilot** uses batch 16, AdamW,
peak LR 2e-5, 50-update warmup, cosine decay to 0.1 peak, and a fixed
0.75 teacher / 0.25 hard-label loss. Score complete GPU FP32 validation at
0/300/600/900; require a >=0.004-BPB endpoint gain versus the verified
1.4098772704 initial score before considering a longer run. Even a passing
pilot would still need a fresh long-run plan, complete compact CPU validation,
three-repeat CPU/RAM/assets audit, and clean-extract reproduction to replace
Stage143. The test split remains untouched. A pilot gain is not a claim that
the 1.35 target has been reached.

## Feasibility result

The one-update Windows preflight passed. Stage105 GPU teacher target log
probabilities differed from the Stage143 cache by at most **9.0599e-6**;
the fixed teacher mixture's maximum log-normalization error was
**9.5740e-7**. One train-only batch-16 forward/backward took **0.8016 s**
after initialization and peak CUDA allocation was **1,471,342,080 bytes**.
All four fixed thresholds pass, authorizing the 900-step pilot. The raw
hash-pinned record is in `../results/stage157-evidence/preflight.json`.
These measurements do not establish pilot quality or inference eligibility.
