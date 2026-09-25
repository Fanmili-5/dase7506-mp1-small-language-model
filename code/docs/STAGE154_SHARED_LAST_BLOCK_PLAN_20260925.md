# Stage154: one heterogeneous final block after seven shared blocks

Stage153 was rejected by its predeclared input-only CPU feature-time gate:
1.250910426x versus an allowed 1.25x. Stage154 is a smaller, materially
different architecture, not a rerun of Stage153: it shares the first **seven**
Stage54 blocks and splits only at block 8. A ends in the original causal
convolution; B ends in causal self-attention. Embeddings, seven-block trunk,
vocabulary head, copy head and deep/future auxiliary projections are shared;
each branch has its own final normalization. The output is the normalized
0.5/0.5 mixture of both branches' prefix-copy distributions. This trades
some diversity for lower CPU and asset cost. Training uses only the supplied
train text and the fixed seed-17 Stage54 sampler/schedule. Validation selects;
test remains untouched until a final freeze.

Before any quality pilot, require synthetic independent-window, causal,
normalized-probability and both-branch gradient tests. Then export a
random-weight OpenVINO two-hidden graph and check validation **inputs only**:
max hidden error <=3e-4, graph bytes +25,000,000 head/count reserve +200,000
source reserve <=64 MiB, and the median of eight interleaved 4-thread CPU
FP32 calls <=1.20x the same Stage143 reference graph. The tightened ratio
reflects the removal of one branch layer; it is a screening proxy, not the
official full-predictor CPU qualification.

Only after passing this screen, train a matched 2,400-step pilot, evaluating
the complete validation set in GPU FP32 at every 300 steps. Continue to a
7,200-step run only if step 2,400 beats Stage54's 1.519950369 BPB by >=0.015.
That is a *continuation* gate, not promotion. Promotion additionally requires
the train-only MKN combination, compact exact inference export, three-repeat
official CPU/RAM/asset qualification and clean-extract reproduction. Stage143
remains protected throughout. If Stage154 does not improve at least ~0.015
early, a less diverse shared-head branch is unlikely to close the remaining
~0.050 gap to 1.35 under these resources.

Risk: two branches with a shared seven-layer trunk may converge to almost the
same predictor; even if feature-time is feasible, dual copy computation may
exceed the official full-predictor 5x CPU limit. Neither gain nor eligibility
is assumed from input-only screening.

## Input-only preflight (2026-09-25)

The two synthetic structural tests passed. On Windows CPU, the random-weight
OpenVINO graph matched PyTorch within `3.82e-6`; its conservative projected
asset size was 60,824,632 bytes; and eight interleaved calls gave a median
feature-time ratio of **1.196239583x** versus the frozen Stage143 graph.
This passes the prespecified 1.20x feasibility screen. It does **not**
establish quality or complete-predictor resource compliance. Exact measurements
and source hashes are in `../results/stage154-evidence/preflight.json`.

## Matched pilot result and decision

The scheduled Windows run finished all 2,400 seed-17 updates, presenting
19,660,800 primary next-token targets in 1,217.59 training seconds. Fixed-file
verification and both structural tests passed. The complete GPU FP32
validation endpoint was **1.513952503 BPB** on 376,599 targets. The archived
same-schedule Stage54 endpoint was **1.519950369 BPB**, a gain of only
**0.005997866 BPB**, below the predeclared **0.015** continuation gate.

Decision: **do not launch a 7,200-step continuation**. The pilot checkpoint
is exploratory; it has not been combined with train-only MKN, exported as a
compact complete predictor, or qualified against the official CPU/RAM/asset
limits. No test score was produced. The entire 300-step curve, source hashes,
checkpoint SHA-256, task status and console transcript are under
`../results/stage154-evidence/`. Stage143 remains the qualified fallback.
