# Stage123: gate-aware calibration screen

Stage94 selected neural vocabulary temperature 1.125 and train-unigram prior
weight 0.0625 for a *fixed* neural/count mixture. Stage114 then attached a
train-fitted confidence gate to that calibrated neural distribution and reached
1.400225 BPB with the cheaper order-five count expert. The confidence gate
depends on the neural top-two log probabilities, so changing calibration may
affect both expert quality and routing. There is no reason the scalar settings
optimal for a static mixture must also be optimal for this dynamic gate.

Keep the Stage92 neural, train-derived order-five count tables, Stage100 gate
coefficients and normalization, anchor 0.0625, gate slope 0.4, and copy-gate
shift 0.25 fixed. Score only the nine predeclared combinations of vocabulary
temperature {1.10, 1.125, 1.15} and train-unigram prior weight
{0.05, 0.0625, 0.075} on complete validation. The center control must
reproduce Stage114 on all 376,599 targets. Validation chooses among these
scalars; no coefficients or model weights are trained on validation.

Only a result below 1.4 BPB warrants exact folded export and full three-repeat
CPU, RAM, and asset qualification. Otherwise record the negative screen. This
is not test tuning; no test score is requested or read during development.
