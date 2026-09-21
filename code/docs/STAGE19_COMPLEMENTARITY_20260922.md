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

## Completed outcome and decision

Task `-b` completed2026-09-21 17:17:28 UTC (September22 01:17 Hong Kong), exit0.
Transport tar SHA256:2a6db4fb38f356b9a8a6b320e8885cfc15a3c007c24d42c9f6f12341fb06d1f5.
Raw JSON and both successful/failed job logs: `results/stage19-evidence/`.
Audit: `results/stage19-audit.json`. Both exact checkpoints are backed up under
the project output folder `outputs/windows-stage19-20260922/checkpoints/`.

| Predictor | CPU validation BPB | Three-repeat CPU ratio | Peak RSS bytes | Inference assets bytes | Decision |
|---|---:|---:|---:|---:|---|
| H |1.482379119|4.802831|1989242880|29678074|new resource-qualified candidate|
| H + .10 counts |1.473335164|5.437347|2008121344|38141585|reject for CPU time|

H CPU scoring seconds:60.181821/60.154874/59.047963; paired baseline
12.400818/12.524879/12.544873. Hybrid seconds:68.956021/67.472113/68.494924;
its baseline12.597122/12.396578/12.691402. Every raw repeat is retained.
The hybrid needs about8.04% lower total scoring time merely to reach5x on this
measurement; practical deployment should seek more margin. This is not a
guarantee of the speedup obtainable by implementation changes.

The auditor verified scoring arithmetic, complete coverage, grouped-loss sums,
all scan source hashes, checkpoint hashes, resource medians/maxima and asset
sizes. It also loaded H, counts and hybrid and checked exact equality of every
neural/count tensor plus config, seed and58982400-target ancestry. No new gradient
targets or count-building pass were used. No test scoring. Qualification is for
this measured Windows CPU, not every machine or a final frozen release.

H diagnostic sampled-train NLL2.663976, validation NLL3.132205, gap.468229.
Compared with F's .598285 gap, row dropout narrows the gap, but the small BPB
gain still does not establish cross-seed robustness. Windows component timings
(first batch32, cached intermediates) were backbone1.122127s, whole head.162255s,
whole forward1.345986s. These separately timed values must not be added or
treated as precise attribution fractions; the backbone is the dominant measured
component, so head-only acceleration has limited overall upside.

Next proposed implementation work, NOT executed in this stage: reduce full-vocab
temporary allocations in train-count queries and avoid redundant log/exp work
in the hybrid while preserving probability semantics. Require full-distribution
causality/normalization and numerical-equivalence checks before resource reruns.
If insufficient, compare a modestly narrower backbone plus statistics against
the current H control under matched training targets. Do not automatically
stack more regularizers, add epochs, or lower the course resource limit.
