# Stage85: direct log of the normalized convex mixture

Stage84 preserves the calibrated predictor and reduces the measured CPU ratio
from 5.1168x to 5.04295x, but remains 0.86% above the hard limit.  Its final
full-vocabulary division only corrects FP32 row-sum drift: the vocabulary,
prefix-copy and MKN components are each normalized before their fixed convex
mixture.

Stage85 directly logs the convex mixture without the redundant final division.
The ideal row sum is exactly one; only FP32 accumulation drift remains.  Export
requires maximum smoke log-normalization error at most `1e-5`, which is 100x
tighter than the fixed evaluator's `1e-3` acceptance check, plus tight
probability/log agreement, complete validation BPB reproduction, and all
resource gates.  Test remains untouched.

## Result

The exact exported checkpoint independently scored **1.4030241332 BPB** on
validation.  Smoke maximum log-normalization error was `9.54e-7`, well inside
both the stage guard and fixed evaluator tolerance.  Three alternating Windows
measurements produced a **4.900107560x** CPU ratio; peak RSS was 2,041,077,760
bytes and conservative assets were 48,570,230 bytes.  All three resource gates
pass.  Checkpoint SHA-256 is
`4d8d5a8356c49287985f0efc02f7dc5dddf33de875cbb4c15eddc394d836cffb`.
Stage85 therefore replaces Stage71 as the qualified validation leader.  Test
was not scored.
