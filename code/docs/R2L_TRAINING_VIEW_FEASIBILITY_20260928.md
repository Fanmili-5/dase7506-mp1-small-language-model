# R2L training-view feasibility review — no experiment authorized

This is a read-only proposal audit, not a predeclared run plan, code change,
course-permission ruling, or validation result. It follows the student's
request to investigate methods before spending GPU time. Stage143 remains
the protected, frozen fallback; no test score is used to select this idea.

## Evidence and competing explanations

Chen et al. (2026), [*Demystifying Training-Time Augmentation for
Data-Constrained Language Model Pretraining*](https://arxiv.org/abs/2606.16246),
reported lower held-out next-token loss for right-to-left (R2L) training
mixed with future-offset prediction than for its plain autoregressive
control. It trained a 150M-parameter model on 75M web-text tokens, with an
altered tokenizer containing direction/offset symbols, and selected over
many epochs. Neither its loss difference nor its training horizon is our
WikiText-2 BPE-2048 BPB. In its study, token replacement combined with
future-offset prediction often interfered; this is relevant because our
Stage54 training already predicts +2/+3 future targets.

Candidate hypothesis: varied prediction direction may regularize the shared
backbone on the small supplied corpus without changing the left-to-right
inference scorer. Rival: splitting updates between directions deprives the
left-to-right path of useful exposure, while the prefix-copy head and
R-Drop/auxiliary objectives may not transfer cleanly across directions.
The local Stage165 embedding-masking failure does not test R2L, but it warns
against assuming all training-only augmentation helps.

## Code-grounded feasibility, not implementation

- `GUIDE.md` fixes the tokenizer/data/evaluator and independent causal
  256-token evaluation windows. Its wording does not explicitly resolve
  whether reversed *training views* of the supplied text count as changing
  the data. This needs instructor or accountable-student interpretation
  before any training. No external text or new tokenizer IDs would be used.
- `code/scripts/train_stage54_hybrid_conv_rdrop.py` samples 259-token spans
  from the supplied train stream, making ordinary +1/+2/+3 labels. A reversed
  view of one such span can construct the same offsets along the reversed
  order, using only train tokens. Per-row direction assignment must be fixed
  across both R-Drop stochastic forwards. Reversal must never enter the
  validation/test loader or evaluator.
- `code/student_structured.py` makes a prefix-copy distribution over only
  the current sequence's causal prefix. On a reversed *training* row, that
  means earlier reversed positions, not illicit evaluation context.
  Structural tests must still establish zero future-prefix influence and
  independent rows for both training directions.
- The paper's new `<r2l>` and offset tokens cannot be copied under the
  fixed-tokenizer rule. A possible adaptation is a zero-initialized,
  train-only 288-wide direction bias applied only to R2L hidden inputs;
  left-to-right inputs remain exactly unchanged at initialization. The bias
  would be stripped on export, preserving the original inference module and
  its graph shape. This is an **untested adaptation**, not reproduction of
  the published method or proof of equal CPU/asset cost after packaging.
- `code/common.py::load_data()` currently loads all three splits, including
  the test text, even when a training runner uses only train/validation.
  Any future candidate runner should use a train/validation-only loader so
  no test text is read during development, and should pin the supplied file
  hashes without test-based selection.

## Why the usual pilot is inadequate

Stage54's 7,200 updates present 58,982,400 primary training targets, about
16.3 times the 3,613,343-token supplied train stream (an exposure ratio,
**not** exact epochs under random-window sampling). The paper's plain AR
minimum occurred around epoch 16, while its R2L+offset minimum occurred
around epoch 44. A 2,400-update gate gives only 19,660,800 target
presentations and could reject a delayed regularization effect without
testing the proposed mechanism. A credible comparison would require the
same longer training-target budget and schedule for a left-to-right control
and an R2L arm, with a complete validation endpoint fixed before either
outcome. That is materially more expensive than reusing the archived
Stage54 control; do not launch it by default under deadline pressure.

If the student chooses this route, first resolve the course-data boundary,
then write a separate pre-outcome protocol with exact direction mechanism,
same-target control, ablation, resource screen, stopping rule, training cost,
and post-training complete CPU-FP32 qualification. A passing validation
score must beat Stage143 and ultimately the <1.35 objective; no test
scoring occurs before a new method freeze. Until those gates are satisfied,
there is no model change and no score claim.

The hypothesis/rival structure was informed by Kassis et al. (2026),
[*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*](https://doi.org/10.48550/arXiv.2609.00065). This is a planning
acknowledgement, not evidence that R2L works for MP1.
