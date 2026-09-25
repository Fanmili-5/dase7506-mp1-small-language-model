# Stage184: fixed training-only block drop-path pilot

## Question, control and stop rule (declared before running)

The protected Stage143 candidate is 1.399686162 complete-validation BPB,
still 0.049686 above 1.35. Earlier Stage22 deep supervision, Stage26 future
prediction and Stage47 R-Drop each improved their matched controls, so removing
those losses is not a promising first test. Stage179's ordinary dropout 0.20
failed; it does not test dropping an entire residual block while retaining
the complete architecture at inference.

Stage184 tests one mechanism and one rate: independent per-example **0.10
training-only drop path** on all eight Stage54 blocks. The retained residual
delta is divided by 0.90. There are no new parameters, buffers, inference
branches, datasets, validation selectors or test-set calls. Unit tests require
identical seed-17 initial weights and bitwise-identical evaluation output
relative to Stage54; source/config hashes are recorded in the run. This pilot
is not a new submission candidate.

Use Stage54's seed 17, batch 32, text/tokenizer, width 288, 8 heads, eight
alternating attention/conv blocks, dropout 0.1, R-Drop and auxiliary losses,
optimizer, and original **7,200-step LR horizon**. Stop at exactly 2,400
updates, or 19,660,800 primary target presentations. Compare only the fixed
step-2400 full GPU FP32 validation endpoint against Stage54's archived
same-step **1.5199503686120217 BPB**. Intermediate 300-step validations are
diagnostics, not selection. The predeclared advancement gate is at least
**0.015 BPB** improvement, i.e. endpoint <= **1.5049503686120217 BPB**.
If it fails, stop Stage184 without a rate/seed sweep. If it passes, authorize
one fixed full training run; that run must then beat Stage143 on complete CPU
FP32 validation and pass CPU, RAM, inference-asset and clean-extract gates
before promotion. Stage143 remains protected. No test scoring before method
freeze.

Strongest objection: the existing dropout and R-Drop already regularize the
small-data model; dropping whole blocks may impede optimization and worsen the
early endpoint. The 2400-step screen will directly test that risk at fixed
train budget rather than relying on a general claim about stochastic depth.
