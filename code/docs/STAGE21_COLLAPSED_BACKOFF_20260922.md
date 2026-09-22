# Stage21: collapse backoff arithmetic, preserve the predictor

## Fixed experiment

Stage20 reached1.473335240 validation BPB but failed CPU at5.046067x.
This stage changes inference arithmetic only: no seed search, new training,
mixture selection, table pruning, test scoring or data changes. Fixed source:
Stage19 .10 hybrid SHA256
77031d3c570e4f4ba9882442ae0189968b5c46783035f9a6d83802b0d8e69046.
Qualified fallback remains H at1.482379119.

The count recurrence q_i=b_i*q_(i-1)+s_i expands to
q_K=unigram*product(b_i)+sum_i(s_i*product_(j>i)(b_j)). Each s_i is sparse.
Compute prefix lookups in descending order, maintain one scale per query, then
add one dense unigram term and the scaled sparse contributions directly into
the neural/copy probability mixture. Avoid allocating a separate dense count
output and multiplying it at every order. Fuse copy scaling with its addition.
All operations use complete2048-token outputs and only the causal input prefix.
Per-order CSR rows have unique token columns; distinct orders are added in
separate operations to preserve collisions. No persistent query cache.

Original inference files remain unchanged. Additive
`student_ngram_collapsed.py` pins Stage20 and the original source chain.
Training uses the unchanged old recurrence and neural path. Inference keeps
the positive-support guard, explicit normalization and final logarithm.

## Acceptance gates fixed before measurement

- Seven synthetic tests: inherited equality, normalization, causality, window
  independence, extreme logits/gates, unsafe-count rejection, unchanged training
  gradients; direct sparse-expansion equality and output-contiguity guard.
- Full validation CPU FP32, four threads, every full-vocabulary output against
  the original Stage19: max logp difference<=2e-5, max normalization error<=2e-6,
  BPB difference<=1e-6;376599 targets/1148007 bytes. Identical serialized tensors.
- Independent official validation, then three fresh-process alternating paired
  Windows baseline/candidate resource runs. Same5x CPU/4GiB RAM/64MiB assets.
  Do not re-run unchanged code until a favorable timing appears.
- Save raw receipts, exact Windows checkpoint, SHA256 hashes and independent
  local audit. No promotion before all gates pass; timing is machine-specific.

AI assistance: algebra, implementation, tests, orchestration and evidence audit.
This is not claimed as a learned-quality gain or a new architecture.

Local full-validation parity passed: max absolute logp error7.6293945e-6,
normalization error7.8324229e-7; old BPB1.4733350659172502 versus
collapsed1.4733352397297033. Same serialized tensors; receipt checked in at
`results/stage21-local-equivalence.json`. Local checkpoint SHA256:
51d64577afb34f4fc40008ffaa78ba033ff0d8565528139192df958d0b5f3563.
Local suite96 tests:91 passed,5 CUDA skipped; all10 fixed-file hashes unchanged.
These are correctness checks, not Windows timing qualification.
