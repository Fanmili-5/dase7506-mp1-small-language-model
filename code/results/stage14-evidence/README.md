# Stage 14 collected evidence

Completed Windows architecture screen, collected on 2026-09-21. All scores
here are complete validation-set CPU FP32 results, not new test results.

The current candidate is B-copy's fixed last-five-checkpoint average:
validation BPB 1.4993342439465787, checkpoint SHA256
`966dcc405ba3d084d06912bdf8136da561e0473dfd119b1ed9f07d223dd9ed0b`.
Its final resource check passed: 3.8306352481954757 times baseline CPU time,
1,975,939,072 bytes peak RSS, and 23,375,787 bytes inference assets.

## Transfer and verification

The complete evidence archive is stored outside the Git repository at project
path `outputs/stage14-results-20260921.zip` (127,296,050 bytes), SHA256:
`49382dff4eae07c0c9530e6e41450135381af5bf7df54469d13d3e1ba0d1f8de`.
Its Mac extraction is `outputs/windows-stage14-20260921/collected/`.
All 52 manifest-listed files were verified, including all six endpoint/average
checkpoint hashes against their score records. The archive includes weights,
window-level NLL arrays, logs, and pinned source files; this Git directory keeps
the smaller evidence subset. Original periodic/resume checkpoints remain on
Windows. The combined three-model evidence archive is NOT a submission bundle.

Remote audit checked checkpoint averaging ancestry and the fixed experiment
protocol. A deterministic Mac reload smoke check of B passed the unchanged
official normalization tolerance (maximum absolute logsumexp residual
3.7923455238342285e-06, official tolerance 1e-3). This is a synthetic-input
contract check, not another benchmark score.

B replaces the old v1 release as the active development candidate. Historical
freezes remain provenance records; the final submission will contain one
immutable code version and its matching predictor/checkpoint bundle. No new
test evaluation was performed during collection.

The measured gain belongs to the complete B recipe: B also uses FP32 output
head computation during mixed-precision training, unlike the historical control.
A precision-matched ablation and paired-seed replication remain outstanding.
