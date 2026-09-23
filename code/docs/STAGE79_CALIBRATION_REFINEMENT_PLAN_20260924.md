# Stage79: final scalar-calibration refinement

Stage78 improved the frozen Stage71 experts from 1.40695954 to 1.40304365 BPB.
Three selected controls were interior, but vocabulary temperature selected the
declared upper boundary of 1.10.  Stage79 therefore performs one final local
expansion and then closes this mechanism.

The fixed `7 x 6 x 6 x 6` grid includes the unchanged Stage71 point, the exact
Stage78 winner, temperatures through 1.20, and half-step neighborhoods for the
other three scalars.  The experts, supplied-train unigram vector, evaluator and
validation split are unchanged.  No gradient target, target-conditioned
feature, external data or test score is introduced.

Whatever the result, no further scalar-grid adaptation follows.  Promotion
still requires exact serialization, independent CPU-FP32 reproduction, and the
unchanged CPU/RAM/asset gates.

## Result

The unchanged Stage71 point reproduced at **1.4069595387 BPB**.  The final grid
selected vocabulary temperature **1.10**, supplied-train unigram-log-prior
weight **0.05**, prefix-copy gate shift **+0.1875**, and MKN weight **0.075**,
scoring **1.4030241589 BPB**.  This is only 0.0000194873 BPB below the Stage78
winner, so the expanded neighborhood confirms that scalar calibration has
converged rather than revealing another large optimization direction.

Scalar calibration is now closed.  Stage80 must serialize exactly these values
and pass independent CPU-FP32/resource qualification; no further scalar search
is permitted.  Raw evidence is in `results/stage79-evidence/`; test was not
scored.
