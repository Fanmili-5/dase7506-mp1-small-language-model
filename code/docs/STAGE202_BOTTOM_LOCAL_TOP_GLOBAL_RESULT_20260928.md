# Stage202 local-first/global-later backbone: rejected

The fixed seed-17, 2,400-step architecture pilot completed on the Windows
RTX 3070 Ti as a local Scheduled Task. The candidate changed only the order
of the same four gated-convolution and four attention blocks. A pre-outcome
synthetic check verified equal parameter count (`8,106,049`), exact training
export, normalized/causal/independent outputs, one finite batch-32 CUDA
update, and 5,543,585,792 peak allocated GPU bytes. This was not a full CPU
inference-resource qualification.

The fixed endpoint's **complete validation BPB was 1.521048958117056** over
376,599 targets / 1,148,007 bytes. The same-seed, same-target Stage54
step-2400 control was **1.519950368612022**. The ordered candidate was
**0.001098589505034 BPB worse**, not 0.030 BPB better as the predeclared
continuation gate required. The first three 300-step comparisons were better
for the candidate, but it lost that lead by step 1,200; the plan expressly
selects the 2,400-step endpoint, so no early checkpoint is promoted after
inspecting the curve. Training presented 19,660,800 primary targets in
595.61 seconds excluding validation; peak CUDA allocated/reserved memory
was 5.651/5.989 GB.

The first foreground launch was interrupted by a VS Code tunnel reconnect
before any scheduled validation point, with no Python process remaining.
Its partial run directory was retained. The unchanged experiment was then
run through an independent one-off Windows Scheduled Task; status was
`completed` and Task Scheduler's last result was `0`. The completed remote
checkpoint SHA-256, `cf388d9341fdd9c13b349007fd95a6961ffe3ffe5a6ed3ea756a24a49103a135`,
matched the metrics record. All 24 recorded training source hashes matched
the tracked local sources on audit.

The Windows runner checked the supplied fixed-file hashes before and after
training, including reading the test file for its SHA-256. It did **not**
tokenize, predict, evaluate, or use test labels for model selection. The
training loader itself opened train and validation only. This distinction is
recorded explicitly rather than claiming the test file was never opened.

The [pre-outcome plan](STAGE202_BOTTOM_LOCAL_TOP_GLOBAL_PLAN_20260928.md)
therefore rejects Stage202. Do not train it to 7,200 steps, export or qualify
its CPU predictor, or score test. The frozen Stage143 checkpoint/graph and
its test record are untouched. Raw preflight, run, progress, complete metrics,
task status and post-run audit are in
[`../results/stage202-evidence/`](../results/stage202-evidence/).
