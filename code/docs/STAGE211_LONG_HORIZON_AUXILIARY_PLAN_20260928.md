# Stage211: long-horizon train-only future-token supervision

Stage143 remains the best resource-qualified model at 1.399686162 complete
validation BPB and its already-frozen 1.415657617 test BPB; neither meets
the student's acceptance levels (<1.35 validation, <1.38 test). Stage208–210
added a jointly trained lexical hierarchy but gained only 0.002391 BPB in
the same-target 2,400-step pilot. A further inference-time lexical head is
not a credible use of the CPU/asset margin.

## Distinct hypothesis and fixed method

The high-loss validation group includes medium-frequency next tokens absent
from their current 256-token causal prefix. The Stage54 backbone already
predicts future offsets 2 and 3 as training-only auxiliary labels, which
mostly emphasize local syntax. Stage211 asks whether representing broader
article context helps generalize to unseen local continuations: retain
the exact Stage54 causal hybrid Transformer/copy architecture, but predict
future training-token labels at offsets **[2, 3, 8, 16, 32, 64]**, sharing
the existing total auxiliary weight **0.2** equally across six heads.
The auxiliary heads are stripped for inference; no additional inference
parameters or operations are permitted. The supplied tokenizer, train text,
validation scorer and 256-token independent evaluation windows are fixed.
Future tokens are training labels only, never inputs to the causal model;
the existing Stage54 recipe already uses future labels at offsets 2/3.

This is not a claim that distant exact-token prediction is inherently useful.
Its strongest objection is label noise: the distant token may be weakly
related to the immediate continuation and dilutes the existing offsets 2/3.
The matched pilot is designed to reject that hypothesis promptly.

## Predeclared gates and comparison

1. Input-only preflight: config differs from Stage54 **only** in the listed
   future offsets. Copy all shared initial weights from a seed-17 Stage54
   control, including the convolution blocks, and restore the RNG after
   building the extra heads. At step zero, the candidate's inference
   distribution must exactly match the control; verify 2,048-way
   normalization, prefix causality and independent rows. One synthetic
   batch-32 BF16 R-Drop update on Windows must have finite gradients,
   peak allocated GPU memory <=7 GiB and reserved memory below GPU total.
   Verify exported inference config/state removes *all* future heads.
2. If preflight passes, process the same seed-17 sampled-start RNG draws
   for the first 2,400 32x256 training windows, AdamW settings, R-Drop
   primary/deep losses and
   exact prefix of Stage54's 7,200-step learning-rate schedule. The
   comparison has **19,660,800 primary next-token targets** in both arms;
   Stage211 additionally presents the declared long-horizon auxiliary
   labels. A draw in the final 64 tokens of the training stream is clamped
   to fit the longest future label; record the count, which means a tiny
   number of primary windows may differ from the archived control even
   though the draw sequence and total primary-target count match. Evaluate
   complete GPU FP32 validation every 300 steps, but
   decide only at the fixed 2,400-step endpoint against Stage54's
   **1.5199503686120217 BPB**. Require >=**0.030 BPB** improvement to
   justify any full 7,200-step continuation. No offset, weight, seed or
   endpoint sweep follows failure.
3. A passing pilot is not a score-qualified model. A full run must beat
   Stage143 and reach <1.35 on complete CPU FP32 validation, then pass
   actual CPU <=5x, RSS <=4 GiB, inference assets <=64 MiB, causality and
   clean extraction. Only a separately frozen final method may be tested.
   Preserve Stage143 and its release bundle unchanged.

This document is a pre-outcome experiment contract, not a result or a
promise that the score gap will close.
