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
