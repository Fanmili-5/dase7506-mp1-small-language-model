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

Deployment archive SHA256:
8eb1d0c6752865a8a16d89023a755d2318b19d614fd4345efb9fc4e812190c71.
Windows one-off task `MP1-stage21-20260922-a` started2026-09-22 08:12:10 UTC
(16:12 Hong Kong); no other Python workload was running at launch. Its seven
synthetic tests ran before full equivalence. Windows parity passed with
BPB1.473335240302804, max logp error5.7220459e-6 and normalization8.3446503e-7.
Windows checkpoint SHA256:
6999e012e74f438f30b39cb36ce69e23df8aa75cd3bd80a422a0b350d3d99d31.
Independent official scoring and resource gate still pending at this entry.

An additional local-only ragged-table regression test passed after launch.
It compares the expanded counts to the old recurrence for missing contexts,
lengths1/3/4/17/256, repeated IDs and scales0/.1/.9, with immutable repeatable
outputs. This synthetic test RNG is not a model training/selection seed.
Running Windows implementation/measurement sources have not been modified.

## Completed Windows measurement

Job completed2026-09-22 08:21:23 UTC. Official validation and all three fresh
resource processes returned BPB1.473335240302804.

| Measurement | Result | Gate |
|---|---:|---|
| Baseline seconds | 12.2451384 / 12.6848014 / 12.3499954 | median12.3499954 |
| Candidate seconds | 60.0980746 / 61.6058403 / 59.8974415 | median60.0980746 |
| Ratio of medians | 4.8662426708 | PASS: maximum5 |
| Peak process RSS | 2,025,037,824 bytes | PASS: maximum4GiB |
| All uncompressed inference assets | 38,153,250 bytes | PASS: maximum64MiB |

This is the first resource-passing implementation of the fixed Stage19 hybrid.
Compared with the prior qualified H, validation BPB is lower by0.0090438782.
The prediction gain was already observed in Stage19; Stage21 supplies an
equivalent implementation that passes the budget on the measured machine.
Timing margin below5x is2.675%; this is narrow and not a universal guarantee.
No repeated unchanged-code measurement was discarded to obtain this pass.

Transport archive SHA256:
36d07ac890f66f6892849782c6027d3f394731325eb97674caff30dc80a48655.
Raw evidence belongs in `results/stage21-evidence/`; exact Windows checkpoint
is backed up in project `outputs/windows-stage21-20260922/collapsed-hybrid.pt`.
The audit checks score arithmetic/coverage, all source hashes, exact tensors and
ancestry, serialized file size, all inference assets and resource aggregation.
Local `results/stage21-audit.json` completed with status `verified` and
`qualified_on_measured_windows_cpu=true`; transport/checkpoint hashes match.
Final local suite97:92 pass,5 CUDA skipped; all10 fixed files unchanged.

This candidate is a development selection, not a final frozen/tested release.
H and all historical candidates remain preserved. No new training/search
targets, no test score, no leaderboard submission in this stage.

## Follow-up priorities

The immediate CPU blocker is resolved. Further quality work should therefore
be separated from equivalent inference optimization: fix a small hypothesis
and comparison budget, select on validation, then recheck the exact exported
predictor. The .10 mixture was best at the boundary of Stage19's0/.05/.10 grid;
a small bounded extension can determine whether that branch still has room,
but a large gain is not established. A more involved alternative is a gate
trained only on training prefixes, compared against this fixed-weight control;
target-dependent seen/unseen diagnostic groups must never be gate inputs.
Neither follow-up is launched by this stage; no claim of reaching1.3 BPB.
