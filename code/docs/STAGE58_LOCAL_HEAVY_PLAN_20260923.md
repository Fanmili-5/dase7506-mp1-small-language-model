# Stage58: local-heavy width reallocation preflight

Stage54 shows that replacing half the attention blocks with gated causal
convolutions both improves validation and admits a wider residual stream.
Stage58 tests a more aggressive resource allocation before any training: width
320, SwiGLU hidden width 640, global attention only in layers 1 and 5, and
gated kernel-7 causal convolutions in the other six layers.  Prefix-copy64,
eight total blocks and the collapsed MKN route remain present.

The reduced number of quadratic attention blocks pays for the wider token,
residual and output representation.  This is an architecture hypothesis, not a
quality claim.  A random-weight complete inference bundle must pass three
alternating repetitions of the unchanged 5x CPU, 4 GiB RSS and 64 MiB asset
gates.  Failure stops the branch before training.  No test split is scored.
