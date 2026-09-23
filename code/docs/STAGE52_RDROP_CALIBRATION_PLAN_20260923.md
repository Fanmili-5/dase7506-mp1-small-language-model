# Stage52: transfer the frozen calibration to R-Drop

Stage47 changes training only and exports the same `student_structured` inference
graph as Stage26. Stage52 therefore applies the scalar settings frozen by
Stage32 before Stage47 existed: vocabulary temperature 1.075, supplied-train
unigram-log-prior weight 0.05, prefix-copy gate shift +0.25 and modified-KN
mixture weight 0.1125.

No Stage47-specific calibration search is performed. This avoids another
adaptive validation grid while testing whether R-Drop's neural improvement
survives the established hybrid predictor. The unigram prior and modified-KN
tables use only supplied training text.

The candidate is serialized directly with the already-qualified fast collapsed
inference arithmetic, scored independently on CPU FP32 validation and measured
for three alternating baseline/candidate repetitions. It advances only if CPU
time is at most 5x, peak evaluation RSS at most 4 GiB and conservative
uncompressed inference assets at most 64 MiB. Test remains untouched.
