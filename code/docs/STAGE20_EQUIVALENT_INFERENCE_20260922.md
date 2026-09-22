# Stage20: preserve the predictor, remove inference overhead

Fixed input: Stage19 .10 hybrid SHA256
77031d3c570e4f4ba9882442ae0189968b5c46783035f9a6d83802b0d8e69046.
CPU validation BPB1.473335164; three-repeat Windows time ratio5.437347 failed.
Qualified fallback H remains1.482379119 at4.802831x. No new training, data,
statistics, mixture weights or architecture search in this stage.

## Implementation

New additive `student_ngram_fast.py`; all old inference sources stay unchanged.
Count-query recurrence is unchanged, but multiplies a call-local dense result
in place and adds only existing sparse edges rather than allocating another
full-vocabulary increment tensor at each order. Buffers are never mutated.
CSR next-token IDs are unique within each row, so sparse writes do not collide.

Eval uses one fused distribution:
`(1-w)*sigmoid(-g)*softmax(logits) + (1-w)*sigmoid(g)*copy + w*ngram`.
It takes a single final log rather than computing neural log-mixture followed
by another log-mixture. This is the same mathematical predictor, not bit-exact
floating-point arithmetic. The positive train-count probability floor is
certified at checkpoint load, including finite nonnegative masses and positive
backoffs, with a conservative bound above16*FP32 tiny. This prevents log(0)
without clipping; unsupported statistics are rejected. Training retains the
original log-space neural computation. No cross-window or target-dependent cache.

## Fixed acceptance procedure

1. Synthetic full-context equality, normalization, extreme gates/logits,
   causality, window/batch independence, unchanged buffers and training-gradient
   tests. Check all official fixed-file hashes.
2. Compare every full-vocabulary output on every validation window, four-thread
   CPU FP32: max absolute log-prob difference<=2e-5, normalization error<=2e-6,
   BPB difference<=1e-6. No tolerance tuning after observing results.
3. Export without changing any tensor; verify serialized tensor equality and
   preserve ancestry/source hashes. Independent official validation scorer.
4. Three fresh-process baseline/candidate resource comparisons on Windows,
   unchanged course limits5x/4GiB/64MiB. Record failure if still too slow.
5. Archive evidence and exact predictor; do not call test or claim deployment
   qualification from Mac timings. No automatic fallback training.

AI assistance: implementation, tests, mathematical equivalence reasoning,
orchestration and evidence audit. This optimizes the existing custom model; it
does not claim a new architecture or a new learned-quality improvement.

Initial local real-checkpoint verification stopped at the first batch: max
log-prob difference3.814697e-6 passed, but normalization error4.289672e-6
exceeded the fixed2e-6 threshold. Added explicit row-sum normalization of the
fused FP32 mixture to correct summation drift. Thresholds remain unchanged.
The initial output directory is retained; retry uses a new directory. This
normalization is mathematically an identity for the ideal normalized mixture,
but its FP32 effect must also pass the same full-output and BPB parity checks.

Local full-validation check passed on the original Stage19 checkpoint:
376599 targets, max absolute log-prob error7.629395e-6, normalization error
7.832423e-7. Old BPB1.4733350659, optimized1.4733352398 (difference1.739e-7).
These Mac results are numerical-equivalence evidence, NOT Windows resource
qualification or learned-quality improvement. Certified positive count floor:
8.301483e-26. All serialized tensors remain identical. Local exported checkpoint
SHA256:6ec4807789009f42ea1d49babc53e0fb6be99ae1f12686cd35b557c200f2fe49.
Local receipt: `runs/stage20-local-equivalence-b/equivalence.json`.

Deployment tar SHA256:f0c53270fc8ea5c0926efdb6eed4214566e8945c33b746a591957e0f6c6ce6e7.
Windows performs its own full-output parity/export, then independent official
scoring and three-repeat resources; no fixed evaluator or historical predictor
source was modified. The recovered private SSH forward uses localhost62267.
