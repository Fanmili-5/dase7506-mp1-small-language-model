# Stage183: arithmetic versus geometric fusion of the frozen experts

## Question fixed before scoring

Stage143 blends its frozen neural/copy distribution `N` and train-derived
order-six MKN distribution `C` with a causal positionwise weight `w`:
`M=(1-w)N+wC`. Its complete validation score is 1.399686162 BPB. The
answer-aware expert oracle is much better but not deployable; an MLP that
changed only the scalar `w` did not transfer across validation halves. A
different question is whether a **tokenwise agreement correction**, rather
than a new context gate, can improve the normalized distribution.

For each causal position, form a normalized geometric expert product
`G(v) ∝ exp((1-w) log N(v) + w log max(C(v), 1e-12))`, then score
`P_beta=(1-beta)M+beta G` for the fixed grid
`beta = {0, 0.1, 0.25, 0.5, 1}`. The floor prevents undefined logarithms
for zero-count vocabulary entries. The formula uses no target or future
token at inference; validation targets are read only to measure NLL. The
two distributions and gate are extracted from the unchanged Stage143
predictor, not refit. Every `P_beta` is fully normalized over all 2,048
tokens, and beta zero must reproduce Stage143 on the entire validation set.

The diagnostic must cover all 376,599 targets and 1,148,007 raw bytes with
the SHA-pinned checkpoint, source and ONNX graph. Report each grid cell and
both contiguous halves of the 1,472 independent windows. **Advance only if**
the best predeclared nonzero beta improves complete validation by at least
0.015 BPB and improves both halves. This is a feasibility gate, not a new
qualified score. A passing cell would still need a causal inference module,
FP32 parity/normalization checks, a complete CPU validation run, three
fresh-process 5x/4GiB/64MiB resource checks and clean-extract reproduction.
No test scoring or changes to Stage143 assets occur in this diagnostic.

Strongest objection: MKN can assign near-zero mass to a correct rare token,
so geometric fusion may amplify the wrong expert's mistakes. If the fixed
grid fails the 0.015 gate, stop this route rather than adjust floors or
search more coefficients on the same validation labels.
