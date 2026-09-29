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

## Complete resource result

The exact exported checkpoint (SHA-256
`c1b5e45b853b34074a215a5d71a9c61e8a96ed7ea0aca082b868b994d3d69e9c`)
reproduced 1.400224947 CPU FP32 validation BPB. Three fresh-process runs
measured baseline median 23.704637 s and candidate median 122.484984 s,
or **5.167132x**. Peak RSS was 2,052,743,168 bytes and conservative assets
50,403,435 bytes; those two limits passed, CPU did not. The short-batch
3.196% gain did **not** persist sufficiently through complete scoring.
Stage122 is not qualified and is not promoted; no test scoring occurred.
Full raw records are in `../results/stage122-evidence/`.
