# Stage213 fixed sliding-attention quality pilot: stopped

Stage213 was a separately [predeclared quality diagnostic](STAGE213_COURSE_ALIGNED_SLIDING_PILOT_PLAN_20260928.md)
for the exact Stage212 architecture. Stage212's stricter internal 1.20x
feature-speed screen had failed at 1.499304x; its complete Windows CPU
score was **not** measured. Stage213 documented that a same-host projection
of the course-style complete CPU ratio was 4.818x, below but close to the
actual 5x course limit. This projection did not reclassify Stage212 as a
resource pass. It only justified testing whether the unchanged architecture
had enough quality upside to warrant a formal resource audit.

The Stage213 synthetic BF16 GPU step passed with finite loss and local
query/key/value gradients, 5,962,258,432 bytes peak allocated memory and
6,230,638,592 bytes peak reserved memory. The subsequent fixed Windows
seed-17 pilot completed **2,400 updates** and **19,660,800 primary next-token
targets** under Stage54's first 2,400 of 7,200 learning rates. All 57
identical named starting tensors matched Stage54. Its one clamped sampler
start matched the Stage54 control's recorded one clamped start. The complete
GPU FP32 validation split contains 376,599 targets and 1,148,007 UTF-8 bytes.

| Step-2,400 complete validation | BPB |
| --- | ---: |
| Stage54 matched control | 1.5199503686120217 |
| Stage213 four sliding-local-attention blocks | 1.5338717702955510 |

Stage213 **regressed by 0.013921402 BPB** and failed the predeclared
`>=0.030` improvement gate. Every recorded 300-step intermediate validation
point was also worse than its same-step Stage54 reference; none is selected
as a model. The route stops without a 7,200-step continuation, trained
OpenVINO export, formal complete CPU/RAM/asset qualification, or any new
test scoring. This rejects the exact seven-position replacement under the
fixed training budget, not all local-attention architectures.

The [synthetic GPU record](../results/stage213-sliding-gpu-preflight-a.json),
[fixed run plan](../results/stage213-pilot-run.json),
[complete metrics](../results/stage213-pilot-metrics.json),
[progress](../results/stage213-pilot-progress.json) and
[Windows job status](../results/stage213-pilot-job-status.json) are retained.
All 32 training-run source hashes matched the local committed files after
collection. The unpromoted checkpoint remains on the Windows host with SHA-256
`de7f109b40cea983368bef7d821d7014f4907e09ac6de6865139726eda207e94`.
Stage143 remains the protected 1.399686 validation / 1.415658 frozen-test
fallback, which still misses the student's score thresholds.
