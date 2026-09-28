# Stage212: fixed-width causal sliding-attention core (pre-outcome)

Stage143's full validation BPB is 1.399686162; its frozen full-test BPB is
1.415657617. Neither meets the student's <1.35 aspiration or <1.38 minimum.
Stage211's distant train-only auxiliary labels regressed at the matched
2,400-step endpoint. Do not tune that branch or use the test split again.

## Problem and single fixed candidate

The Stage54 backbone alternates four global-attention blocks with four
kernel-seven gated depthwise-convolution blocks. A convolution shares the
same positional kernel for every local context. Stage212 replaces exactly
those four local blocks (layers 2/4/6/8) with **causal seven-token sliding
attention**, retaining the other four global blocks, width 288, eight heads,
depth eight, context 256, tokenizer, output/copy mechanism, auxiliary losses,
R-Drop recipe, optimizer and training target count. No offsets, windows,
seeds, head widths or local-layer subsets will be swept after results.

The local block may select different nearby tokens according to content;
unlike Stage206 it does not discard global exact attention, and unlike
Stage199 it does not add a second simultaneous mixer. The strongest
objection is that the adaptive local mixer is redundant with four global
layers or too expensive on CPU. This is an architectural hypothesis, not a
prediction of a 0.05-BPB gain.

## Predeclared gates

1. Synthetic input only: every local position may attend to itself and at
   most six preceding positions; no future position and no other row.
   Verify finite normalized log-probabilities, prefix causality, row
   independence, full-vocabulary inference export identity and finite
   local-query/key/value gradients. The candidate and Stage54 control must
   share every identical named tensor at initialization; only local-mixer
   tensors may differ. Verify all shared names and shapes explicitly.
2. Windows FP32 OpenVINO input-only screen: convert a random-weight feature
   model at fixed 256-token context; require eager/compiled hidden maximum
   error <=3e-4, candidate/reference Stage143 feature median <=1.20 on eight
   interleaved batch-32 four-thread measurements, and conservative assets
   <=64 MiB including any added checkpoint parameters and source reserve.
   This is only a feasibility screen, not full scorer qualification.
3. If step 2 passes, one synthetic Windows BF16 R-Drop update at physical
   batch 32 must have finite loss/gradients, peak allocated memory <=7 GiB,
   and reserved memory below GPU total. A failed screen stops before data.
4. Only then run one seed-17, 2,400-update validation-only pilot using the
   same sampled training-window draws, batch 32, primary 19,660,800 targets,
   and first 2,400 learning rates of Stage54's 7,200-step schedule.
   Decide only at step 2,400 complete GPU FP32 validation against Stage54's
   1.5199503686120217 BPB. Require >=0.030 BPB gain to license a separately
   fixed full run. Earlier checkpoints are diagnostic, not a selection grid.
5. Even a passing pilot is not a score-qualified model. A full run must
   improve complete CPU FP32 validation toward <1.35, pass official CPU
   <=5x, RAM <=4 GiB, assets <=64 MiB and clean extraction, then be frozen
   before any full test. Preserve Stage143's checkpoint, graph and records.

This plan is committed before its input-only measurements and contains no
observed Stage212 result.
