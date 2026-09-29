# Stage208: jointly trained lexical hierarchy (pre-outcome plan)

The student's requested acceptance remains complete-validation BPB <1.35 and
complete-test BPB <1.38. The protected Stage143 scores 1.399686162 on
validation and its already-frozen test score is 1.415657617. This experiment
does not select on test and cannot be called complete by passing a pilot.

## Problem and mechanism

Stage151/207 locate much error on unseen current-window continuations, but
the frozen spelling/prefix residuals in Stage160/207 regress. The proposed
change is **joint training**, not another frozen-backbone head fit: retain
the Stage54 hybrid attention/convolution R-Drop backbone and ordinary
vocabulary/copy branches, but add a normalized lexical hierarchy predicting
the next fixed BPE token. The 2,048 token strings from the supplied fixed
tokenizer are lexicographically sorted and recursively split at the median,
giving a fixed 11-decision binary tree. Each internal decision is a learned
linear function of the current causal hidden state. The leaf probability is
the product of its 11 branch probabilities. The hierarchy's probability is
mixed with the ordinary vocabulary softmax by a learned current-state gate,
then with the existing within-window copy branch. This shares parameters
among nearby token spellings without changing token IDs, data or scorer.
Tree construction uses the supplied tokenizer only; neither validation nor
test text or labels enter it. There is no external text, pretraining, future
token, cross-window state, lookup from validation/test or network call.

The strongest objection is that lexical clustering can impose the wrong
conditional geometry and duplicate information in the plain head. The fixed
pilot is meant to reject it quickly. This is structurally different from
Stage14's generic mixture of softmaxes and Stage207's frozen prefix residual,
but the negative evidence from both lowers our expectation.

## Fixed control, preflight and stopping gates

1. Synthetic input-only checks: complete 2,048-way log normalization within
   1e-5; no future-position or other-row effect; exact 11-decision paths;
   finite train loss/gradients. On Windows RTX 3070 Ti, batch-32 BF16
   R-Drop optimizer update must fit below 7 GiB allocated and GPU total
   reserved. A conservative FP32 inference-asset projection must fit 64 MiB.
   Eight warmed four-thread synthetic 32x256 lexical-head runs must have
   median <=0.50 seconds (the Stage143-to-5x total headroom is about 33
   seconds for 46 validation batches, or 0.72 seconds per batch). This is a
   conservative screen, not actual full CPU qualification.
2. If feasible, run exactly Stage54's first 2,400 seed-17 updates, same
   sampled training targets, batch 32, optimizer and 7,200-step LR prefix.
   The only added trainable structure is the lexical head/gate. Compare the
   complete GPU FP32 validation endpoint to the archived same-budget Stage54
   endpoint 1.5199503686120217. Require at least 0.030 BPB improvement to
   justify full 7,200-step training. Record source/data/config/checkpoint
   hashes and both fixed validation halves. No post-hoc tree, seed, gate
   initialization or learning-rate grid follows a failure.
3. A passing pilot still is not a replacement: full CPU FP32 validation
   must beat Stage143 substantially and meet <1.35, and the exact deployed
   checkpoint must pass CPU <=5x baseline, peak RAM <=4 GiB, uncompressed
   assets <=64 MiB, clean extraction and causality. Only freeze the method
   before any new test run. Keep the Stage143 checkpoint/bundle unchanged.

The feasibility and quality gates are predeclared decisions, not outcomes.
