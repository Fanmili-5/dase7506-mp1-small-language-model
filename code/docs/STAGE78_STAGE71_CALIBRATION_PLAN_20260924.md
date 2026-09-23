# Stage78: bounded calibration of the Stage71 experts

Stage71 gains through neural/count specialization but deploys the same scalar
post-training calibration inherited from earlier branches: unit vocabulary
temperature, no supplied-train unigram adjustment, no prefix-copy gate shift,
and one global MKN weight.  Earlier Stage31--32 evidence found a 0.00244 BPB
gain from these four scalar controls on a weaker frozen pair, so the mechanism
is worth retesting once on the current leader.

Stage78 freezes the exact Stage71 averaged neural expert and Stage25 MKN expert.
It evaluates one declared `7 x 7 x 7 x 9` validation grid covering vocabulary
temperature, a train-unigram log-prior coefficient, prefix-copy gate shift and
count-mixture weight.  The unchanged Stage71 point is included as an exact
reference.  All tensors remain frozen, the only non-model vector is derived
from supplied training text, and no setting uses target identity at inference.

This is a diagnostic gate, not a promoted checkpoint.  A positive result must
be serialized with the exact selected scalars, independently reproduced by the
fixed CPU-FP32 evaluator, and pass the unchanged CPU/RAM/asset limits before it
can replace Stage71.  Test remains untouched.

## Result

The unchanged Stage71 point reproduced at **1.4069595387 BPB**.  The fixed grid
selected vocabulary temperature **1.10**, supplied-train unigram-log-prior
weight **0.05**, prefix-copy gate shift **+0.25**, and MKN weight **0.075**,
scoring **1.4030436461 BPB**.  This is a 0.0039158925 BPB gain with no new
gradient target or inference asset.  Prior weight, gate shift and mixture
weight are interior points, while temperature selects the declared upper
boundary.

One final local expansion around the selected neighborhood is therefore
permitted.  It must include the unchanged Stage71 point and the Stage78 winner;
after that scan scalar calibration is closed.  Stage78 itself remains a
diagnostic result until exact serialization and resource qualification.  Raw
evidence is in `results/stage78-evidence/`; test was not scored.
