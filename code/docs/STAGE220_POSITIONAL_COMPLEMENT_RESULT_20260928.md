# Stage220 result: complement is not concentrated at window starts

The [pre-outcome plan](STAGE220_POSITIONAL_COMPLEMENT_PLAN_20260928.md)
used only the two SHA-pinned Stage162 validation true-target streams. The
analysis reproduced Stage143 at **1.399686162**, Stage155 at
**1.409877270**, and their uniform full-position mixture at
**1.376182664 BPB** over all **376,599 targets / 1,148,007 bytes**.
The eight disjoint position bins sum exactly to the complete target count
and the complete 18,702.6220-nat mixture gain. No test data was read.

The first 64 of 256 positions supplied only **22.3865%** of the total
mixture gain, below the predeclared **50%** concentration requirement.
Their fixed 50:50-mixture-only policy scored **1.394424562** complete
validation BPB, a **0.005261600** gain versus Stage143, below the
predeclared **0.012** requirement. First-32 and first-128 policies scored
1.397244432 and 1.388599708 respectively. The later 32-position bins
contributed roughly 13% each, so the known complement is broadly
distributed and not primarily a short-prefix effect.

**Decision:** both gates fail. Do not engineer, train or deploy a
short-context helper based on these frozen experts. The diagnostic
position policies still contain two large inference models and fail the
asset/CPU budget; they are not course scores or legal submission models.
This result does not rule out a genuinely new compact predictor, but it
removes the measured rationale for an early-window-only one. Stage143
remains the sole protected qualified candidate and the student's score
objective remains unmet.

Exact bin counts, gains, fixed-policy BPBs and source hashes are in
[`../results/stage220-evidence/positional-complement.json`](../results/stage220-evidence/positional-complement.json).
