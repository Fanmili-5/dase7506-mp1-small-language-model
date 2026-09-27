# Stage202: local-first, global-later core backbone (pre-outcome plan)

This is a new, validation-only architecture experiment. It does not alter the
frozen Stage143 candidate and must not load or score test.

## Distinct mechanism and controlled comparison

The accepted Stage54 backbone alternates four global RoPE-attention blocks
at layers 1/3/5/7 with four kernel-7 gated causal-convolution blocks at
2/4/6/8. The Stage202 candidate uses the **same eight blocks and widths**,
but orders the four local-convolution blocks first (1/2/3/4) and the four
global-attention blocks last (5/6/7/8). The causal receptive field of the
stacked local blocks can build token/word-scale features before global
aggregation. This is a structural hypothesis, not a claim of improvement.
Prior Stage145 changed the attention/conv *count*, and Stages153/197 changed
the final two-block branch topology; none measured this fixed four-plus-four
ordering against the alternating control.

Hold fixed the Stage54 config except `conv_layers`, model implementation,
seed 17, batch 32, 256-token windows, sampled-window RNG, R-Drop,
deep/future training losses, AdamW, first 2,400 learning rates of the
7,200-step Stage54 schedule, and complete FP32 validation evaluator. The
same-target control is Stage54 at step 2,400: `1.5199503686120217 BPB`.
Select the **fixed 2,400-step endpoint only**; intermediate 300-step scores
are diagnostics, not checkpoint selection. Continue to an independently
planned 7,200-step run only if endpoint BPB is at most
`1.4899503686120217`, a gain of at least 0.0300. This deliberately high
gate reflects the much larger gap to the student's target. No seed or
layout sweep follows a failed pilot.

## Admission and later gates

Before the pilot, assert exact config diff, equal trainable parameter counts,
causal prefix/row independence, normalized probabilities, finite gradients
and one batch-32 CUDA BF16 optimizer step within 7 GiB allocated memory.
The training/validation loader verifies only the supplied train and
validation files. Archive source/data hashes and the full validation at the
fixed endpoint. If this pilot passes, a new complete run must beat the
frozen Stage143 complete validation and then earn its own CPU-FP32 <=5x,
peak RAM <=4 GiB and uncompressed inference-assets <=64 MiB measurements.
Nothing here authorizes another test run; a new method freeze is required.
