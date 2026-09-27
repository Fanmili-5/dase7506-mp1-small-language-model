# Stage194: one-base causal residual expert (predeclared, validation only)

The user wants complete-test BPB below 1.35, with 1.38 the minimum acceptable
delivery score. The current single tested candidate scores 1.415657616535609.
This plan does not use that test result to choose architecture settings or
hyperparameters. Its development baseline is Stage143's complete-validation
1.399686162042141 BPB; no further test access is permitted during development.

## Problem and alternatives considered

Stage143+Stage155's 50:50 mixture gave 1.376182664 on validation, but two
complete neural models breach the inference-asset limit and likely the CPU
limit. A larger single model did not match the mixture. Small changes to
count weights, tokenizer masks, and output gates have yielded negligible gains.
Possible distinct routes are: compact two-expert inference; a trained causal
mixture gate; a residual expert over Stage143 features; a new from-scratch
Transformer family; or a longer schedule of the current model. We prioritize
the residual expert because it reuses the single qualified feature computation
and can represent context-dependent corrections without a second full model.

The architecture is fixed before measurement: freeze Stage143/Stage105 base,
add one width-288 eight-head causal attention block, width-576 MLP, and
rank-64 output correction. The final projection is zero-initialized, so the
untrained predictor should be identical to the base. No future labels,
cross-window state, external data, or evaluator changes are allowed.

## Feasibility gate before training

Run only synthetic input on Windows CPU: checkpoint hash and source check,
zero-init parity (maximum log-probability deviation <= 1e-5), causal
future-token perturbation (prefix deviation <= 1e-5), finite normalized
probabilities, and exact expert state byte count. Reject if projected
uncompressed assets (55,810,412 base bytes + expert state bytes + 262,144
bytes source reserve) exceed 67,108,864 bytes. Interleaved CPU median
candidate/base time must be <= 1.20. This is a conservative screen, not a
replacement for the official full validation resource qualification.

If the gate passes, use the supplied train split only for gradients and the
supplied complete validation split for selection. A matched pilot must compare
the same initialized expert and the same train-token draws under (1) hard
next-token NLL and (2) 50:50 Stage105+Stage155 teacher distillation with hard
NLL, keeping the base frozen. Fix seed, steps, batch, learning-rate schedule,
teacher/hard weights, and validation times before launching. A pilot is worth
continuing only if the teacher arm beats both Stage143 and the hard-only arm
by at least 0.015 and 0.010 BPB respectively on full validation. A passing
pilot is not a deployable result; the full CPU/RAM/asset gates and independent
replication remain necessary. No complete test run or submission is authorized
by this plan.

Failure hypothesis: complementary information from Stage155 may not be
recoverable from the frozen Stage143 hidden states; a residual could simply
overfit train prefixes or add too much CPU latency.
