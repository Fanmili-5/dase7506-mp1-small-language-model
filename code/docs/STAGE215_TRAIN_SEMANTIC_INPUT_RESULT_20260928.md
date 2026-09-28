# Stage215 result: train-derived semantic input helps early, misses endpoint gate

The fixed Stage215 pilot completed on the Windows RTX 3070 Ti with no task
error. Its train-derived 2,048-by-64 basis was built only from the supplied
3,613,343 training-token stream. A second build on the same Windows host
reproduced basis SHA-256
`9db3fd3f63c61ecba9871f1d1d27fc1972aa12c0f26c69291809696ad353a1f7`.
The Mac and Windows low-rank decompositions differ at the byte-hash level;
the trained checkpoint stores its basis, so this is a training-portability
caveat, not a claim of cross-platform bit-identical retraining. The fixed
train/tokenizer hashes matched the repository.

The predeclared synthetic Windows four-thread CPU screen passed. Zero-start
Stage54/candidate log probabilities were identical; maximum normalization
error was `4.77e-7`. Four-pair median candidate/control time was **1.00839x**,
and projected Stage143-plus-semantic assets were **56,608,428 bytes** below
64 MiB. Three structural tests passed on Mac and Windows. These are input-
only feasibility checks, not complete trained-predictor CPU/RAM qualification.

The scheduled pilot then processed exactly **2,400 updates and 19,660,800
primary next-token targets**, with seed 17, batch 32 and Stage54's exact
7,200-step learning-rate prefix. Its fixed endpoint complete GPU FP32
validation was **1.503975623068176 BPB** over 376,599 targets and 1,148,007
bytes. The same-step Stage54 control was **1.519950368612022 BPB**, a real
matched gain of **0.015974746 BPB** but below the preregistered **0.030**
advancement gate. At steps 300 and 600 the gains were about 0.04029 and
0.03634 BPB, then declined to 0.02124 by step 1800. This is consistent with
faster early learning rather than a demonstrated superior final model; it is
not proof of the exact mechanism.

The task status was `completed`, Windows Scheduled Task exit result was zero,
and an independent SHA-256 of the saved checkpoint matched the metrics:
`f8335fc2903e0ae6b3df31813d271f5578ed6553a775b407480f6e28d8fe123d`.
All **27** source/config/development-data hashes matched this checkout.
Training took 593.63 seconds excluding 15.83 seconds of validation; peak GPU
allocation/reservation was 5.651/5.985 GB. These are pilot costs only.

**Decision:** stop Stage215 at the prespecified endpoint. Do not select an
earlier checkpoint, alter co-occurrence span/rank/scale, continue to 7,200
steps, export a CPU model or score test. Stage143 remains the only protected
resource-qualified candidate at 1.399686 complete validation and 1.415658
already-frozen test BPB, still above the student's 1.35/1.38 requirements.
Raw [preflight](../results/stage215-semantic-preflight-a.json),
[run](../results/stage215-evidence/run.json),
[progress](../results/stage215-evidence/progress.json),
[metrics](../results/stage215-evidence/metrics.json) and
[terminal job status](../results/stage215-evidence/job-status.json) are retained.
The failed checkpoint remains on the Windows experiment host, outside the
Stage143 inference bundle.
