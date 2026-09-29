# Stage216 result: causal relative-distance bias misses advancement gate

The Stage216 plan was committed before the Windows quality pilot. This
architecture adds 32 zero-start, causal distance buckets per head to the four
global attention blocks of the unchanged Stage54 hybrid. It tests whether
learned recency routing repairs context errors spread across validation
targets. It does **not** change tokenizer, data, training budget, or target
availability. The supplied test split was not opened for Stage216.

The Windows random-weight preflight passed its limited checks: exact
zero-start log-probability parity (maximum error 0), causal prefix error 0,
FP32 OpenVINO hidden-state error `3.96e-6`, eight-pair candidate/reference
feature-time ratio `1.007991`, conservative asset projection `56,654,419`
bytes, and two synthetic BF16 updates within the 8-GiB GPU (`5,958,008,832`
bytes peak reserved). These are **input-only feasibility measurements**, not
trained-predictor CPU/RAM qualification.

The fixed seed-17, batch-32 pilot completed 2,400 updates and 19,660,800
primary train-target presentations. The exact first 2,400 learning rates of
the 7,200-step Stage54 schedule were used. At the fixed endpoint, complete
GPU FP32 validation was **1.5177762488452309 BPB** over all **376,599
targets / 1,148,007 UTF-8 bytes**. The same-step Stage54 control was
**1.5199503686120217 BPB**, a gain of only **0.0021741197667908 BPB**.
The predeclared advancement requirement was a gain of at least **0.030**;
the pilot failed it by a wide margin. Earlier checkpoints are not selected.

The Windows scheduled task reported `completed` and exit code `0`; its
checkpoint independently re-hashed to
`67523507624004a91888d319cb3580ce2f5745a4d9c64f110c7ae17acb4573d2`,
matching the metrics JSON. All **26** recorded source/config/development-data
SHA-256 values match the committed Mac tree. Training took 662.478 seconds,
with 16.461 seconds of validation; GPU peak allocation/reservation were
5.650/5.989 GB. These are training measurements, not a resource pass.

**Decision:** stop Stage216. Do not perform a 7,200-step continuation,
trained-model CPU/RAM qualification, or test scoring. The protected Stage143
still scores 1.399686162 on complete validation and 1.415657617 on its
already-frozen complete test, above the student's <1.35 and <1.38 thresholds.
This negative result rejects this fixed learned-distance-bias mechanism; it
does not establish that all relative-position architectures fail.

Evidence: [preflight](../results/stage216-evidence/preflight.json),
[metrics](../results/stage216-evidence/metrics.json),
[run](../results/stage216-evidence/run.json),
[trajectory](../results/stage216-evidence/progress.json), and
[job status](../results/stage216-evidence/job-status.json).
