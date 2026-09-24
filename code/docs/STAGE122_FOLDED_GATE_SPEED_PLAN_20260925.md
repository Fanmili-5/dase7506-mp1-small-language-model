# Stage122: exact folded-calibration dynamic-gate speed probe

The Stage115 order-five dynamic gate scores 1.400225 validation BPB but its
single-repeat CPU ratio is 5.170. The Stage121 static folded export shows that
the fixed Stage94 temperature, unigram prior, and copy-gate shift can be
absorbed into the existing affine parameters without retraining. Stage116
shows that caching count-row maxima saves 1.37% on one CPU batch. Stage122
combines those two *exact-prediction* transformations and folds the gate's
feature standardization into four fixed coefficients.

First compare the full Stage115 and transformed Stage122 probabilities on a
fixed validation-input batch without labels, and time ten warmed CPU FP32
forwards. Only if the median is at least 3% faster and the probability error
is at most 3e-6 should the candidate be exported for complete validation and
the three-repeat resource audit. Otherwise stop with a negative speed probe.
No new training, new fitted values, validation-label fitting, or test scoring.

## One-batch gate result

The fixed CPU FP32 batch passed equivalence (maximum probability error
1.82e-6) and the ten-repeat median fell from 2.685419 to 2.599588 seconds,
or **3.196% faster**. This meets the preregistered 3% threshold, so the exact
checkpoint export and complete validation/three-repeat resource audit proceed.
The probe record is in `../results/stage122-evidence/probe.json`.
