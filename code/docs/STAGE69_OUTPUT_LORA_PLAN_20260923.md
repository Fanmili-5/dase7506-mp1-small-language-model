# Stage69: frozen-backbone low-rank output adaptation

The byte-composed continuation and vocabulary intercept are reproducibly
positive but together improve the neural model by only 0.00101 BPB. Stage69
tests a more expressive zero-start mechanism: a rank-16 residual on the
vocabulary projection, trained while the entire Stage67 backbone, copy route,
embedding matrix and output bias remain frozen.

The low-rank residual starts at exactly zero (Gaussian A, zero B), so the initial
predictor is identical to Stage67. It receives five deterministic full passes
over supplied training text in independent 256-token windows. The fixed
candidate is the epoch-3/4/5 parameter average. At export, the residual is
materialized into one independent output matrix; inference still performs one
vocabulary matrix multiplication, while the input embedding remains unchanged.
This adds about 2.25 MiB of assets but no new deployed matmul.

Only validation is scored. The mechanism advances to MKN/resource qualification
only if the fixed exported average improves Stage67 materially. Test remains
untouched.
