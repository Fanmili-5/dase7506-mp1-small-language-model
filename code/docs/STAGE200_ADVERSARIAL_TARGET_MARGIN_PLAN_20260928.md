# Stage200: train-only adaptive target margin (pre-outcome plan)

This is a development-only experiment. It must not read, score, or tune on
the test split. The frozen Stage143 checkpoint, graph, source and test record
remain untouched. No release candidate is implied by a pilot.

## Hypothesis and fixed comparison

Wang, Gong and Liu, *Improving Neural Language Modeling via Adversarial
Training* (ICML 2019; https://arxiv.org/abs/1906.03805), derive a target-output
embedding perturbation equivalent to subtracting
`alpha * ||w_target|| * ||h||` from the true-class logit during training. This
pilot adapts that margin to the **final normalized prefix-copy distribution**;
it is not an exact reproduction of their plain-softmax method. The target
margin's two norms are detached, as in the paper's stable implementation.

Use `alpha=0.005` fixed from the paper, no alpha search. Same Stage54 model,
seed 17, random windows, batch 32, learning-rate schedule (full 7200-step
schedule), R-Drop, deep/future objectives, and 2400-step pilot endpoint. The
only changed term is each of the two primary NLLs: apply the target-logit
margin to the final predictive distribution before computing target NLL.
The R-Drop KL remains on the unperturbed distribution. There is no inference
graph, checkpoint schema, parameter, or runtime change.

Matched prior control is Stage54 at step 2400, complete validation
`1.5199503686120217 BPB` on the same tokenizer and scoring protocol. Before
any outcome, require pilot BPB <= `1.4899503686120217` (at least 0.0300 BPB
gain) to consider a full 7200-step run. This unusually high gate reflects
the student's minimum score of 1.38 on a held-out test and the observed
Stage143 test/validation gap; a tiny early gain is insufficient. If the
pilot fails, stop; do not tune alpha or seed locally. Even if it passes,
resource qualification and a new freeze decision are required before any
new test. Course staff should clarify whether a second frozen candidate may
replace the already test-scored Stage143 submission.

## Safety checks

The pilot loader verifies and opens only train and validation files. Before
training, test the target-margin identity, alpha-zero equivalence, finite
gradients and one CUDA update. Record source hashes, elapsed time, GPU
memory, and full validation at the fixed endpoint. Do not compare against
any test outcome for model selection.
