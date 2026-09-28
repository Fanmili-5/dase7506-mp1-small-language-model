# Stage210 matched quality result: stop the lexical hierarchy route

The Windows `b` task completed the fixed seed-17 **2,400-step**, batch-32
BF16 R-Drop pilot after the Stage210 FP32 OpenVINO head-only feasibility
gate passed. It processed **19,660,800 primary training targets**, following
Stage54's exact 7,200-step learning-rate prefix. The model differs from
Stage54 only by the jointly trained, fully normalized lexical-tree head and
its gate. The supplied tokenizer, train/validation data and evaluator were
unchanged. The loader opened only train and validation; no new test score was
computed.

At step 2,400, the **complete GPU FP32 validation** result over **376,599
targets / 1,148,007 UTF-8 bytes** was **1.517559228901995 BPB**. The
archived same-seed, same-target Stage54 control was **1.5199503686120217**,
so the fixed gain is only **0.002391140 BPB**. This is much less than the
predeclared **0.030** early-quality advancement gate. The advantage was
similarly small at 600/900/1,200/1,500/1,800/2,100 steps, so there is no
endpoint-only anomaly to explain the decision. The result is not close to
the student's <1.35 complete-validation target.

The job status is `completed`, its final progress record agrees with the
metrics, and the saved checkpoint SHA-256
`a13127c4b63539b593cefa766904391010917fa8a0ddcc476f1125e5c88b1d41`
was independently re-hashed on Windows. All **28** listed source/config/
development-data hashes match the Mac repository. Training took 750.193 s
and validation 18.689 s; GPU peak allocation was 5.921 GB. These are pilot
training measurements, not a full CPU/RAM deployment qualification.

**Decision:** stop Stage208–210. Do not run a full 7,200-step continuation,
CPU/RAM release qualification or new test. Stage143 remains the best
resource-qualified candidate at 1.399686162 complete-validation BPB, still
above the requested acceptance threshold. The Stage210 random-weight
OpenVINO preflight graph is not an inference bundle.

The first `a` task stopped before training because an already-committed
plan file was missing from Windows. Its failure is retained separately; the
`b` task used a new run directory and did not overwrite evidence. Raw
records: [preflight](../results/stage210-openvino-tree-a-result.json),
[metrics](../results/stage210-pilot-metrics.json),
[run](../results/stage210-pilot-run.json),
[progress](../results/stage210-pilot-progress.json),
[b status](../results/stage210-pilot-job-status.json), and
[a status](../results/stage210-first-launch-status.json).
