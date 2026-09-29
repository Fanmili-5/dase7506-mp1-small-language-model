# Stage132: transfer the exact local successor cache to Stage115

Stage43 found that a strictly causal within-window exact successor cache improved
the older Stage26 predictor by 0.001686 BPB, but the gain was too small to
justify a third inference expert then. Stage115 now sits at 1.400224946 BPB
with its train-fitted five-order MKN gate, though it exceeds the CPU limit.
This validation-only transfer screen asks whether the *same* local mechanism
is complementary to the stronger neural/count model. It changes neither
training data nor any checkpoint weights.

Use the existing source-pinned `scan_stage43_exact_local_cache.py` on the exact
Stage115 exported checkpoint. The **primary setting** is fixed from Stage43:
minimum exact matching context order 2, maximum order 8, and cache mixture
weight 0.10. The script emits its historical full grid for diagnostic
comparison; choosing a new grid optimum after viewing validation would be
exploratory, not confirmatory. Require Stage115's zero-weight control to
reproduce 1.400224946 BPB on all 376,599 validation targets. Only a primary
gain of at least 0.001 BPB and BPB below 1.4 warrants a normalized inference
implementation, train-only parameter selection, causality/unit tests, and a
three-repeat official CPU/RAM/asset qualification. Stage115 already needs
about 3.3% CPU speedup, so a quality gain alone is not a deployable win.

The cache may read only predecessors in the current independent input window;
it resets for every row and never sees target or future tokens when building
the table. Targets are used solely for scoring the fixed distribution. No
test split is scored. Preserve Stage85 as the qualified fallback.

## Fixed-setting screen result

The zero-weight control reproduced Stage115 at **1.400224973 BPB** on all
376,599 validation targets. The prespecified order-at-least-2, weight-0.10
cache scored **1.398978671 BPB**, an improvement of **0.001246302 BPB**;
it was active on 45,770 targets (12.1535%). The same fixed setting also
ranked first in the emitted exploratory grid, but its confirmatory status
comes from being selected in Stage43, before this Stage115 result.

This passes the diagnostic quality gate, **not** the submission gates.
Stage115 alone failed its CPU limit at 5.170x in its one-repeat preflight,
and this diagnostic has not implemented a full-vocabulary inference cache or
measured CPU/RAM/assets. The next step is a causal normalized predictor and
one-batch performance pilot; only then is exact full validation and resource
qualification justified. Raw scan evidence is in
`../results/stage132-evidence/scan.json`. No test split was scored.
