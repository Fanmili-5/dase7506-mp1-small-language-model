# Stage156: test whether the rejected larger neural model adds distinct errors

Stage155's fixed five-checkpoint average scored 1.4098772704 complete GPU
FP32 validation BPB, worse than the qualified Stage143's 1.3996861620.
The previous Stage99 screen shows that an order-six count expert improved the
Stage92 neural-only predictor from 1.4063126973 to 1.4012884355, about
0.0050 BPB. That historical gain cannot be assumed to transfer to Stage155,
and Stage155's 47,960,933-byte feature graph plus the roughly 20-MB count
component would threaten the 64-MiB asset limit. Do not combine them by
default.

Instead, make one **diagnostic-only** comparison of Stage155 and the existing
Stage143 prediction errors on the complete validation target stream. Verify
both checkpoint hashes, all 376,599 target positions, and the independently
known BPBs. Score one prespecified 50:50 probability mixture and partition
the target NLL by the existing Stage151 train-frequency / causal-prefix groups.
The target-derived groups cannot be used as inference gates. The ensemble
itself is over the resource budget and is not a submission candidate.

Only if the fixed 50:50 mixture reaches <=1.385 validation BPB **and** gains
>=0.014 BPB over Stage143 should a train-only distillation pilot be considered.
This is a value-of-information threshold: the earlier Stage91 teacher reached
1.383202 neural-only, while Stage92 transferred only ~0.0012 BPB to the
deployed student. A weaker teacher is unlikely to close the 0.049686 gap to
1.35. Otherwise, archive the error comparison and pivot to a genuinely new
conditional output/training objective instead of another ensemble weight scan.
No test scoring or checkpoint selection follows this diagnostic.

## Observed complete-validation diagnostic

The exact fixed 50:50 mixture scored **1.3761826642 BPB** on all 376,599
validation targets, a **0.0235034979-BPB** gain over Stage143 and below the
predeclared 1.385/0.014 advancement thresholds. Both individual target
streams reproduced their independent full-evaluator scores to <2e-12 BPB.
The four train-frequency/prefix groups and their additive NLLs are in
`../results/stage156-evidence/complementarity.json`.

This is evidence of complementary *errors*, not a resource-qualified model:
the two checkpoint/graph bundles exceed the 64-MiB asset cap, the diagnostic
uses a cached validation target stream, and no CPU/RAM qualification was run.
The passed gate permits a **train-only distillation pilot**, conditional on
a concrete teacher implementation and GPU memory preflight. It does not
authorize selecting ensemble weights or reporting 1.376 as a submission
score. The 1.35 target remains unmet.
