# Stage31: coarse scalar calibration screen

The fixed Stage26 neural and Stage25 modified-KN experts were screened with four
legal scalar settings: vocabulary temperature, a supplied-train unigram log
prior, a prefix-copy gate shift and the count mixture weight. The 3x3x3x3 grid
used validation only for selection; it introduced no learned validation tensor,
new gradient target, external data or test score.

The unchanged Stage30 point reproduced at 1.4488983671 BPB on CUDA FP32. The
best coarse point, temperature 1.05, unigram-prior weight 0.05, copy-gate shift
+0.25 and count weight 0.125, scored **1.4467608924 BPB**, a 0.0021374747 gain.
Temperature, prior weight and gate shift all selected the positive boundary, so
the result remains diagnostic rather than frozen. Stage32 performs one bounded
expanded-neighborhood scan that includes both this point and the unchanged
Stage30 reference. No repeated adaptive fine search follows Stage32.
