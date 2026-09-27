# Stage204 causal pair-input architecture: quality gate failed

The fixed seed-17, 2,400-step pilot completed on the Windows RTX 3070 Ti.
It changed Stage54's input representation by adding one trainable,
8,192-bucket hashed **previous/current-token** embedding at fixed scale 0.5,
and kept its architecture, sampled-window seed, batch 32, optimizer,
7,200-step learning-rate prefix, auxiliary losses, and R-Drop recipe. It
presented 19,660,800 primary training targets in 596.54 training seconds.
All 25 recorded training source hashes matched the tracked local sources,
the final checkpoint SHA matched the Windows file, and Task Scheduler
reported successful completion (last result 0).

At the predeclared step-2,400 endpoint, the candidate scored **1.509304977670925
BPB** on the complete 376,599-target / 1,148,007-byte validation set.
The same-target Stage54 control was **1.5199503686120217 BPB**: a real
**0.010645390941097-BPB** gain. But the predeclared continuation gate
required at least **0.030 BPB**, and the early advantage had mostly shrunk
by the fixed endpoint. Stop here; do not select a better-looking early
checkpoint, change the hash/buckets/scale or run a fresh full schedule.
No CPU-final predictor, formal resource qualification, method freeze or new
test score followed.

Input-only preflight-b had passed after an exactly equivalent integer-export
repair: maximum OpenVINO/eager hidden error `3.69549e-6`, 65,497,409-byte
conservative projected assets (<64 MiB), candidate/reference feature-time
ratio `0.983568`, and peak synthetic allocated GPU bytes `5,551,885,824`.
These checks prove only that the pilot architecture was feasible to train and
export with random weights, not that a final checkpoint satisfies the CPU,
RAM or asset limits. The original failed preflight-a, export diagnosis,
successful preflight-b, complete pilot plan/progress/metrics, and the
[pre-outcome plan](STAGE204_HASHED_BIGRAM_INPUT_PLAN_20260928.md) are retained
under [`../results/stage204-evidence/`](../results/stage204-evidence/).

Stage143 remains the protected best resource-qualified Windows candidate:
complete validation 1.399686162 and previously frozen full-test 1.415657617
BPB. That full-test result remains above the student's 1.38 minimum. Stage204
never opened or scored test through its preflight or train/validation loader.
