# Stage207: explicit causal word-prefix conditioned residual

The frozen Stage143 candidate remains at 1.399686162 complete-validation
and 1.415657617 already-frozen complete-test BPB. The student's objective
is <1.35 on validation and minimum acceptable test result is <1.38. Stage206
found that substituting two attention blocks with rank-16 linear memory
regressed against its matched control, so rank/layer tuning of that route is
not justified.

## Problem-first evidence, and what it does not prove

The Stage191 validation diagnostic identified 300,020 positions with an
ASCII current-word prefix supported by the supplied-train prefix table. Of
those, 35,977 true following tokens were **not** in the corresponding train
prefix's observed successors. A new read-only decomposition of the frozen
Stage143 target-probability stream shows these 35,977 positions have mean
target NLL 5.794766 nats and contribute 0.261994 of its 1.399686 BPB.
This group is defined using the true target and is **not available as an
inference gate**. It supplies a failure mode, not a projected improvement.
Even removing 0.05 BPB only from this group would need roughly 1.105 nats
lower mean NLL per member; a small residual is therefore a risky hypothesis.

Stage191's train-only prefix-count mixture worsened full validation, while
Stage160's spelling-composed output residual did not explicitly read the
current observed word prefix and also regressed. Stage207 combines their
missing roles: a deterministic causal character representation of the
current ASCII word prefix with the frozen Stage105/143 Transformer hidden
state, plus spelling-shared output rows. No new backbone is trained and no
validation/test labels enter inference.

## Fixed model and comparison

Use the exact frozen Stage105 checkpoint and its Stage143 train-only count
predictor as in Stage160. At each position, derive an active ASCII word
prefix from the input token IDs only, resetting at the beginning of every
256-token window and on non-letter tokens. Keep at most 32 letters, encode
the last 16 as 52 case-sensitive character IDs with zero left-padding.
A 32-dimensional character embedding and one fixed 128-dimensional MLP
combine with the frozen 288-dimensional causal Transformer hidden state.
The rank-128 residual uses Stage160's fixed-token spelling-composed output
rows, is zero-initialized, and is active only where a word prefix exists.
The 2,048-way distribution is renormalized after adding the residual.

One seed `207017`, 2,400 updates, physical batch 16, FP32 AdamW,
learning rate 1e-3 with 50 warmup steps and cosine decay to 1e-4,
weight decay 0.1, and residual dropout 0.25 are fixed before training.
Only supplied training tokens update the residual; Stage105/count weights
remain frozen. This is compared to its exact zero-start Stage143 complete
validation control, not to a differently trained model. The terminal
2,400-step checkpoint alone decides the pilot gate; intermediate scores
are diagnostic and cannot be selected after inspection.

## Feasibility and stop gates

1. Synthetic/input-only preflight: zero-start output error <=2e-6,
   normalized probabilities <=1e-5, future-prefix and independent-row
   differences <=1e-6, projected total assets <=64 MiB, and eight-pair
   four-thread prefix-feature+head median <=0.25 seconds per 32x256 batch.
   This is **not** full trained-predictor CPU/RAM qualification.
2. If preflight passes, run the fixed train-only pilot. Step zero must
   reproduce Stage143's 1.399686162 BPB within 2e-5 over all 376,599
   validation targets. The step-2,400 endpoint must improve complete
   validation by >=0.020 BPB and improve both prespecified 736-window
   halves; otherwise stop, without rank/seed/dropout sweeps.
3. A passing pilot still requires exact compact CPU FP32 integration,
   trained full-validation reproduction, three-repeat CPU<=5x,
   peak RAM<=4 GiB and uncompressed assets<=64 MiB, followed by a new
   method freeze before any new test scoring. Protect Stage143 throughout.

This plan is a falsifiable candidate, not a claim that a word-prefix head can
reach 1.35 or that the retrospective 35,977-target group is identifiable
without labels at inference. AI assistance in design and execution must be
disclosed and understood by the student.
