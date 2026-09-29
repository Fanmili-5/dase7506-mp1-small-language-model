# Stage187: locate a single-operator Linux CPU optimization opportunity

This is a **validation-input-only runtime diagnostic** for the unchanged
Stage143 graph, not a new model or score. The protected Stage143 checkpoint,
ONNX graph, source files, tokenizer and scorer remain unchanged. No test
tokens are passed to the profiling script.

## Why this diagnostic

Stage170 measured a Linux one-thread ratio of 5.502555x, requiring about
9.13% reduction in complete candidate time to reach 5x on that host if the
baseline stays fixed. Stage171 attributed 15.255/18.148 seconds of its first
eight validation batches' model inference to the frozen feature graph;
Stage172's ONNX Runtime backend swap improved only feature time by 5.31%,
below its predeclared 12% gate. Blindly changing the count table or backend
is therefore not a justified next step. The remaining question is which
compiled graph operations actually consume the Linux CPU time.

## Fixed measurement before observing results

On the same class of Ubuntu x86-64 runner, install pinned CPU dependencies.
Verify checkpoint/graph/source hashes and fixed course files. Compile the
**same** graph on OpenVINO CPU FP32, one thread, one stream, pinning disabled,
once with `PERF_COUNT` and once without. Use the first eight independent
full-sized validation input batches from the official window generator.
Warm both compiled graphs before timing. Require identical FP32 shape and
max hidden-state difference `<=1e-3` on the first input. Measure each
profiled batch wall time and collect OpenVINO node `real_time`, `cpu_time`,
node type and execution type; aggregate by type and preserve the top nodes.
Also time the unprofiled reference on the same batches to estimate profiling
overhead. Do not calculate BPB, modify predictions or read labels.

Node times may not add to process wall time, especially with fusion, and
profiling changes runtime; use them to identify a *candidate mechanism*, not
to claim a resource pass. Advance to **one** algebraically equivalent
inference-only graph rewrite only if the profile reveals a clear source-local
operation that plausibly saves at least 12% of feature time without adding
assets beyond 64 MiB. The rewrite must first pass exact-shape, finite,
causality/normalization and full-distribution parity checks, then a paired
feature-time gate of >=12% before any full-validation or resource promotion.
Otherwise stop this route and retain the reported Linux risk.

## Observed result and decision

The [Linux run](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36191286223)
completed successfully. The archived [raw node profile](../results/stage187-linux-evidence/stage187-linux-node-profile.json)
pins the unchanged checkpoint, graph and source hashes. Both compiled
versions returned exactly equal FP32 hidden tensors on the first input
(`max difference = 0`). There were eight batches, 256 independent input
rows, **zero scored targets** and no test score.

The unprofiled reference took **14.790237 s** and the profiled graph
**14.742509 s** over those eight batches; the small reversal is ordinary
measurement noise, not a speedup claim. Summed OpenVINO node times were
14.312990 s. `FullyConnected` nodes accounted for **9.884000 s**, or
66.83% of the unprofiled reference wall time. The eight MLP input
projections alone contributed **4.477477 s**. The next categories were
`Subgraph` 1.347242 s, `Concat` 1.012093 s and `Slice` 0.443730 s.
Individual `Concat`/`Slice` changes cannot plausibly save the required
~12% of feature time even if their measured cost vanished entirely.

The dominant cost is spread across 32 dense projections, not one redundant
operator. Replacing FP32 matrix multiplication with a faster kernel is a
backend/hardware project rather than a simple algebraically equivalent
graph rewrite; Stage172's tested alternate backend gained only 5.31% in
feature time. No concrete single source-local rewrite currently meets the
predeclared plausibility gate. **Stop Stage187 without modifying inference
or promoting a new model.** This does not prove all graph optimization is
impossible; it keeps the Linux one-thread risk explicit while preserving
the Windows-qualified candidate and the submission timeline.
