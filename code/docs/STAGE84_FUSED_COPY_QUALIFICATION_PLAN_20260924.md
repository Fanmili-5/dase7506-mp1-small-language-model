# Stage84: fused-copy qualification

Stage83 preserves the calibrated 1.40302417-BPB predictor without runtime
temperature work, but its measured CPU ratio is still 5.1168x.  Profiling of
the same hybrid family shows that the output/mixture head is a material part of
the complete forward pass.

The existing copy path scatters attention into a new dense
`batch x time x vocabulary` tensor and then reads that tensor again to add it to
the vocabulary probabilities.  Stage84 instead scales the attention by the
causal copy gate and scatters it directly into the caller-owned vocabulary
distribution.  It also reuses a non-persistent causal mask.  This removes a
dense allocation, zero-fill and readback without changing parameters or the
probability formula.

Exact-output tests and a direct Stage83 smoke comparison precede independent
validation and the unchanged CPU/RSS/asset gates.  Test remains untouched.

The first smoke run showed `1.19e-7` maximum probability error and `4.77e-7`
normalization error, but FP32 addition reordering amplified one extremely small
probability to a `2.30e-4` log-probability difference.  The acceptance guard
therefore retains tight probability and normalization bounds, permits at most
`3e-4` log error, and still requires complete validation BPB agreement within
`2e-6` before any resource claim.

## Result

The fused implementation reproduced validation BPB `1.4030241741` and reduced
the CPU ratio from `5.116791501x` to `5.042948349x`.  Peak RSS was
2,040,107,008 bytes and conservative assets were 48,567,812 bytes.  The speedup
is real but still 0.86% above the hard CPU limit, so Stage84 remains
unqualified and advances only to removal of the redundant dense normalization.
