# Stage83: norm-folded calibrated qualification

Stage80 exactly serialized the Stage79 scalar calibration but narrowly missed
the CPU limit because it divided every hidden vector by temperature at runtime.
Stage82 removed the division by cloning an untied output matrix, but the extra
matrix increased memory traffic and failed at 5.2685x baseline.

Stage83 instead folds `1 / temperature` into the final normalization affine
parameters.  The tied vocabulary projection then receives the exact calibrated
hidden state with no runtime division and no extra output matrix.  The copy
query, key and gate weights are multiplied by temperature so their inputs are
algebraically restored to the pre-calibration hidden state.  A direct output
equivalence test is required before independent validation and resource gates.

This is an inference-only reparameterization of the frozen Stage80 predictor.
It introduces no targets or selection and does not score test.

## Result

Stage83 reproduced validation BPB `1.4030241743`, used 48,565,110 bytes of
assets and 2,039,779,328 bytes peak RSS.  Its three-repeat CPU ratio was
`5.116791501x`, so it failed the hard time gate despite exact algebraic
reparameterization.  The checkpoint is retained as the tied-weight source for
further kernel optimization, not as a qualified replacement.
