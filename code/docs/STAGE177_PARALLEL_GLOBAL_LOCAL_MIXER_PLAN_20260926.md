# Stage177: parallel global/local mixing in one Transformer backbone

## Problem-first rationale

Stage143 scores 1.399686162 complete validation BPB, leaving 0.049686 to the
aspirational 1.35 goal. Stage146 found high loss for targets absent from the
current prefix; Stages145/150 showed only small early gains from replacing
local convolution with more attention or adding plain depth. Stages175–176
showed that sophisticated routing of the existing neural/count experts does
not reliably transfer across validation halves. The next question is whether
the **same token-mixing block** can preserve short-range convolutional
patterns *and* gain global attention, rather than swapping one for the other
or attaching a single late branch (Stage154).

Candidate: keep Stage54's eight-layer width-288 backbone, four original
attention blocks, four original gated causal-convolution blocks, tied 2,048-
way vocabulary/copy head, R-Drop and train-only auxiliary objectives. Add a
parallel RoPE causal-attention branch to each of the four conv blocks, summed
with the original conv output before its existing residual dropout and FFN.
The branch output projection starts at zero; all original model parameters
and the post-construction global RNG stream must match Stage54 exactly.
Thus the untrained candidate computes the identical function, while training
may learn complementary global mixing at each local stage. The inference
graph has no count expert yet; final MKN must be rebuilt/qualified only if
the neural candidate shows a material gain.

## Prespecified gates (before outcomes)

1. Structural tests: at seed 17, shared tensors and complete step-zero eval
   probabilities exactly match Stage54; output is normalized, causal and
   independent across windows. Train-only/inference exports preserve the
   additional attention weights and remove auxiliary heads.
2. Windows CPU FP32 input-only preflight: export a random-weight feature graph
   and compare with eager PyTorch within 3e-4 maximum hidden difference.
   Require graph bytes +25,000,000 bytes reserved for a compact checkpoint,
   MKN and heads +200,000 source bytes <=64 MiB. Eight interleaved batch-32
   feature calls must have candidate/reference median <=1.25 against the
   exact Stage143 graph. These screens are not a complete predictor audit.
3. If the preflight passes and a single batch-32 GPU update fits the RTX
   3070 Ti, run one seed-17 2,400-update pilot on supplied training text.
   Preserve Stage54's sampled windows, batch 32, 256-token context, optimizer
   and the **first 2,400 steps of its 7,200-step LR trajectory**. Score all
   376,599 validation targets every 300 updates. Compare the endpoint to
   Stage54's same-step 1.519950369 BPB. Require >=0.015 BPB improvement for
   a separately planned full run. The early gate is a cost filter, not a
   guarantee of final <1.35.
4. Only a full candidate beating Stage143 may advance to train-only order-six
   count rebuilding, complete CPU FP32 validation, three-repeat same-host
   <=5x CPU / <=4 GiB RAM / <=64 MiB asset checks, and clean-extract QA.
   No test scoring before method freeze.

Strongest objection: the new global branch may cost CPU without improving
small-data generalization. The first preflight screens runtime; the matched
pilot compares quality at equal training-target presentations. A negative
result stops this route rather than prompting a head-count or seed grid.

## Input-only Windows result and stop decision

Local structure tests passed: seed-17 shared tensors, RNG state and zero-start
predictions matched Stage54 exactly; attention projections received gradients;
the exported inference view preserved causality, normalization and identical
training-view predictions. The Windows random-weight FP32 OpenVINO graph had
**9,266,113 neural parameters** and occupied **36,172,843 bytes**. Maximum
eager/OpenVINO hidden-feature difference was **3.81e-6**, below 3e-4. The
conservative graph-plus-reserve estimate was **61,372,843 bytes**, below
64 MiB.

Eight interleaved batch-32 feature calls had medians **1.788710 seconds**
for Stage177 and **1.243395 seconds** for the exact Stage143 reference,
a **1.438569x** ratio. This fails the predeclared **<=1.25x** CPU-feature
screen by a large margin. The projected asset and parity gates pass, but
the combined pilot-admission gate **fails**. No 2,400-step pilot, complete
predictor qualification, checkpoint promotion or test score follows.
The random graph remains on the Windows experiment host; its SHA-256 and raw
timings are archived in `../results/stage177-evidence/preflight.json`.
The feature-only timing cannot be presented as a full 5x resource ratio.

This mechanism is stopped as specified, rather than shrinking the number of
parallel branches post hoc to make the preflight easier. Stage143 stays
protected at 1.399686162 validation BPB.
