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

## Complete validation result and stop decision

The Windows CPU FP32 run covered all **1,472 independent windows**, **376,599
targets** and **1,148,007 raw bytes**. Beta zero reconstructed the exact
Stage143 score at **1.399686162101 BPB**, within `5.9e-11` of the archived
reference. The largest full-distribution arithmetic reconstruction error was
`1.19e-7`; maximum expert and geometric row-normalization errors were each
`3.58e-6`. The source, checkpoint and graph SHA-256 values matched the
predeclared identities. Both fixed halves contained 736 windows.

| beta | Complete validation BPB | Gain over Stage143 |
| ---: | ---: | ---: |
| 0 | 1.399686162 | 0 |
| 0.1 | 1.399273702 | 0.000412460 |
| 0.25 | 1.398764568 | 0.000921594 |
| 0.5 | **1.398247514** | **0.001438648** |
| 1 | 1.399459656 | 0.000226506 |

The beta-0.5 cell improved both fixed halves by 599.83 and 544.95 nats,
but its full gain is about **one tenth** of the predeclared 0.015 gate and
only about **2.9%** of the remaining 0.049686-BPB distance to 1.35. The
route therefore stops: no deployed geometric branch, new inference asset,
resource claim or test score. This is a target-readout diagnostic, not a
qualified 1.398248 submission candidate. The full JSON and source hash are
in `../results/stage183-expert-fusion.json`.
