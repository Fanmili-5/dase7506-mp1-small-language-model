# Stage19: measured CPU bottlenecks and local-statistics complementarity

Plan fixed before observing new results. No new gradient training or test scoring.
The Stage18 H exported average is fixed at SHA256
`2be8f4f4ac195593835aaa74f042b0b5d0f5473a46eb6d09fba99b3c11d263f7`.
H CPU validation BPB is 1.4823791185117505; its exact weights have not yet had
a final resource check. The qualified fallback remains Stage15 F, 1.485094298.

## Questions and bounded procedure

1. Profile H on Windows CPU FP32, four threads, first full validation batch32.
   Time backbone, vocabulary projection/log-softmax, copy attention/scatter,
   mixture operations, complete head and complete forward separately. Cached
   intermediate microbenchmarks are diagnostic, not additive time attribution
   and never a submission-resource measurement. No implementation changes yet.
2. Reuse the unchanged Stage16 train-only counts (SHA256
   `b1898559c73ccf62e6e945c1a9269e6fe2b0ee4499b5a1d88e7e30020c194230`).
   Fixed mixture weights: 0, .05, .10. No count tuning, new data or learned
   validation parameters. Full validation selects the smallest BPB; retained
   model computes full distributions using only input prefixes.
3. Report seen/unseen-target loss groups for all three settings. These groups
   are answer-conditioned diagnostics, never predictor or gate inputs.
4. Independently score the selected serialized hybrid with the official CPU
   evaluator. Run three fresh-process resource comparisons for H and, if a
   nonzero mixture wins, the hybrid. All assets count toward64MiB, RSS<=4GiB,
   CPU<=5x baseline. No .003 quality trigger in this diagnostic stage: small
   improvements receive a resource measurement, not a statistical claim.
5. Run unchanged train/validation generalization diagnostics on H. Record
   all inputs/source hashes and results, including budget failures. No new
   architecture training automatically follows this screen.

Historical Stage18 .003 advancement trigger and outcomes stay unchanged.
This separate stage tests complementarity, not a retroactive Stage18 success.
If mixtures fail resources, prioritize profiling-guided implementation work;
if they fail quality, do not assume a more elaborate gate will rescue them.

The scan is an explicit version of the existing Stage16 script; it leaves all
historical sources unchanged and adds a pinned new reference, count hash guard,
smaller fixed grid and diagnostic grouping. Substantive AI assistance includes
analysis, implementation, tests and orchestration; disclose it in the report.

## Execution

Windows task `MP1-stage19-20260922-b` started2026-09-21 17:01:11 UTC
(September22 01:01 Hong Kong). Original attempt `-a` failed before scoring
because the profiler assumed a source-hash key absent in native checkpoints.
The profiler now checks the actual module against the pinned StructuredLM hash;
a regression test covers the native schema. Failed logs are retained. No model
weights or historical inference code changed. Windows three new tests pass;
local full suite83:78 passed,5 CUDA skipped. A real F checkpoint profiler smoke
also passed on Mac; its timings are not substituted for Windows results.

The independent serialized-hybrid CPU validation has now reproduced
1.4733351642965178 at mixture weight.10 (H alone1.4823791185117505).
Weight.05 scored1.4742628391522223. These are validation selection results;
resource comparisons are still pending at this checkpoint in the work log.
The .10 mixture increases seen-target loss by837.423 nats but decreases
unseen-target loss by8034.039 nats; net improvement7196.616 nats. This is
diagnostic evidence of complementarity, not a gate allowed to inspect targets.

Added `audit_stage19_results.py` independently checks full-score arithmetic,
coverage, group sums, source/asset receipts and aggregation of three raw resource
repeats. A synthetic test rejects corrupted score arithmetic and false budget
pass flags. Local suite now84:79 pass,5 CUDA skipped; the three startup tests,
not this later local-only audit test, were run on Windows before launch.

Stage18 completed evidence archived at `results/stage18-completed-evidence/`.
Transport tar SHA256:1b808de9c7e1935c05e17ddda5e5a9e63126c10fa22fdeb720cd8d77f7a153c7.
All its pinned sources still match; score arithmetic, checkpoint receipt links,
376599-target coverage and117964800 total gradient targets checked. This check
does not claim to independently reconstruct Stage18's averaged tensors.
