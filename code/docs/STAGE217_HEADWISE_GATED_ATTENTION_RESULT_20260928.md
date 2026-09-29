# Stage217 result: headwise post-attention gate below quality gate

Stage217 tested a query-conditioned, head-specific gate **after** global
softmax attention, inspired by [Qiu et al., *Gated Attention for Large Language
Models*](https://arxiv.org/abs/2505.06708). Our `2*sigmoid` zero-start
variant is not a literal reproduction. The cited paper's large-scale gains
are not evidence of an improvement on this tiny fixed-corpus assignment.
The Stage217 mechanism, matched training and >=0.030-BPB continuation gate
were committed before the quality pilot.

Windows input-only preflight passed: zero-start FP32 log-probability error
`0`, causal-prefix error `0`, normalized log-probability error `4.77e-7`,
random-weight OpenVINO hidden-state error `3.59e-6`, eight-pair CPU feature
time ratio `1.006351`, and conservative asset projection `56,069,362`
bytes. Two synthetic BF16 batch-32 training updates completed with finite
gradients and `6,018,826,240` bytes peak GPU reservation. This is not a
trained full-predictor resource qualification.

The fixed seed-17 pilot completed **2,400 updates** and **19,660,800**
primary training-target presentations. It used the exact first 2,400
learning rates of Stage54's 7,200-step schedule and opened only supplied
train/validation text. At the fixed endpoint, complete GPU FP32 validation
was **1.5149825863073896 BPB** over **376,599 targets / 1,148,007 UTF-8
bytes**. Its same-step Stage54 control was **1.5199503686120217 BPB**,
so the gain was **0.0049677823046321 BPB**, far below the preregistered
**0.030** continuation threshold. Earlier validation points are not
substituted for the endpoint.

The Windows scheduled task ended `completed` with exit code `0`. Independent
checkpoint SHA-256 was
`ad356a442aa3d52f9c46072335b1832b05e7aaf80bfd5fe7283090d3f01d6b4e`,
matching the metrics. All **26** recorded source/config/development-data
hashes matched the committed Mac tree. Training/validation took
`605.065`/`16.218` seconds; GPU peak allocation/reservation were
`5.733`/`6.048` GB. No Stage217 test split was opened.

**Decision:** stop Stage217 without full 7,200-step training, trained CPU/RAM
qualification, or test scoring. The protected Stage143 remains the best
resource-qualified candidate at 1.399686162 complete-validation BPB and
1.415657617 on its already-frozen complete test—above the student's
<1.35/<1.38 thresholds. The negative result is about this fixed small-scale
gate, not a claim that all gated attention fails.

Evidence: [preflight](../results/stage217-evidence/preflight.json),
[metrics](../results/stage217-evidence/metrics.json),
[run](../results/stage217-evidence/run.json),
[trajectory](../results/stage217-evidence/progress.json), and
[job status](../results/stage217-evidence/job-status.json).
