# Stage32: final scalar calibration result

The one bounded expanded-neighborhood scan included the unchanged Stage30 point
and the positive neighborhood selected by Stage31. It evaluated 1,080
combinations of four scalar settings on the complete validation split. The
train-derived unigram vector and both frozen experts were unchanged; no gradient
target, external data or test score was introduced.

The selected settings are vocabulary temperature **1.075**, unigram-log-prior
weight **0.05**, prefix-copy gate shift **+0.25**, and modified-KN mixture weight
**0.1125**. CUDA FP32 validation BPB is **1.4464619984**, improving the
unchanged Stage30 point by 0.0024363687 BPB. All four values are interior to the
declared grid. Relative to Stage31's coarse best, the additional gain is only
0.0002988940 BPB, so scalar calibration is closed rather than repeatedly refined.

This is a diagnostic score until the calibrated arithmetic is serialized,
independently reproduced on CPU FP32, collapsed for efficient count inference,
and passed through the exact resource gate. Stage27 remains the qualified
fallback. No test data was scored.
