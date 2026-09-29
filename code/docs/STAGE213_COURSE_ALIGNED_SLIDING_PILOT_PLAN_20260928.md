# Stage213: course-limit-aligned quality diagnostic of the fixed Stage212 core

Stage212's seven-token sliding-attention architecture passed exact causal,
normalization, eager/OpenVINO parity and projected asset checks, but its
random-weight feature time was 1.499304x Stage143, above the preregistered
1.20x *internal feature-screen* gate. Stage212 therefore stopped as stated;
it has no trained score. The course's actual constraint is complete CPU FP32
scoring <=5x the baseline, not a 1.20x feature ratio. This is a transparent
separate decision to measure quality before paying for an end-to-end resource
audit, not a retroactive declaration that Stage212 passed its old gate.

On the same Windows host, the Stage143 full CPU FP32 median was 86.051106 s
against a 23.786126 s baseline. The paired input-only feature medians were
1.243451 s (Stage143) and 1.864310 s (sliding attention) per batch; there
are 46 batch-32 calls for the 1,472 independent windows. A simple fixed-
head projection is 86.051106 + 46*(1.864310-1.243451) = **114.610638 s**,
or **4.818x** baseline. This is only a plausibility calculation: graph time,
Python overhead, memory effects and a trained graph may differ. The 5x rule
is **not passed** until the actual complete scorer is measured after training.

## Fixed decision path

1. Keep the exact Stage212 source/config: width 288, depth eight, four
   global and four seven-position local-attention layers. Copy all 57 shared
   seed-17 Stage54 tensors; only local-mixer-specific tensors start anew.
2. One synthetic BF16 CUDA R-Drop update at physical batch 32 must have
   finite loss/gradients, peak allocated GPU memory <=7 GiB and reserved
   memory below GPU total. This screen opens no data. Failure stops here.
3. If passed, one seed-17, 2,400-update pilot uses the same Stage54 sampled
   starts, batch 32, primary 19,660,800 targets, offsets [2,3], R-Drop,
   AdamW and exact first 2,400 learning rates of the 7,200-step control.
   Complete GPU FP32 validation at 300-step intervals is diagnostic; only
   the step-2,400 endpoint decides. Require >=0.030 BPB improvement over
   Stage54's 1.5199503686120217 to consider a separately fixed full run.
   No window/width/seed/offset or score-selected checkpoint sweep follows.
4. A passing pilot would still not meet the student's score target or the
   course resource rules. A full run must beat Stage143 and aim for <1.35
   complete CPU FP32 validation, then pass three actual CPU/RAM/assets
   measurements, clean extraction, and method freeze before test. Preserve
   Stage143 throughout. This plan does not authorize another test or any
   public submission.

The stronger objection to this branch is that adaptive local attention may
provide no quality gain over the cheaper convolution. The endpoint gate
is designed to reject it without a long training commitment.
