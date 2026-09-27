# Stage206: bounded causal linear-memory core pilot

Stage143's complete validation/test BPB remain 1.399686162/1.415657617,
above the student's 1.35 target and 1.38 minimum. Stage204/205 changed
adjacent-token input representations and did not reach their fixed +0.030
early-quality gate. Stage202 changed block order, while Stage177/199 added
parallel local/global paths. None tested a different causal global mixer.

## Mechanism and fixed comparison

This is a new **core**, not a head residual or a seed search. Keep the Stage54
eight-block R-Drop recipe, seed 17, training sampler, optimizer and exact
target budget. Replace only softmax-attention blocks 3 and 7 with low-rank
positive-feature linear attention: each 16-dimensional key forms a cumulative
key/value state over the *current* 256-token window; queries normalize by the
matching cumulative key mass. Blocks 1 and 5 retain ordinary attention; the
even blocks retain causal convolution. No state crosses window or row bounds.
The tokenizer, train text and evaluator are unchanged. No test data is read.

The problem-first hypothesis is that a persistent compressed within-window
state may encode distant topical/lexical evidence more economically than
four independent softmax-attention blocks, leaving CPU room for a stronger
core. The strongest objection is that a rank-16 positive kernel loses sharp
content retrieval; the fixed pilot is designed to reject it quickly.

## Predeclared gates

1. On synthetic data only, verify finite normalized probabilities, exact
   first-position/window reset, prefix causality and independent rows. Export
   the random-weight FP32 feature graph. PyTorch/OpenVINO hidden parity must
   be <=3e-4 for batch 1 and 32. Conservative inference-asset projection
   must be <=64 MiB. Eight interleaved four-thread feature calls must be no
   slower than 1.25x the Stage143 feature graph. One BF16 batch-32 R-Drop
   gradient update must have finite loss/gradients and peak allocated GPU
   memory <=7 GiB. Failure stops without a training run.
2. Only after preflight passes, run the same first 2,400 updates of Stage54's
   7,200-step schedule, seed 17, physical batch 32 and complete validation
   endpoint. Compare to Stage54's same-schedule 1.5199503686120217 BPB.
   Require a gain >=0.040 BPB to justify full 7,200-step training. This
   stringent gate reflects the ~0.05 complete-validation gap; no seed,
   rank, layer or weight sweep follows a failure.
3. Even a passing pilot is not a replacement. A full run must beat Stage143
   on complete CPU FP32 validation, then pass actual full-predictor CPU<=5x,
   RAM<=4 GiB and assets<=64 MiB on Windows plus clean extraction. Freeze
   the method before any new test scoring. Keep Stage143 untouched.

This document predeclares the decision; it contains no quality result.
