# Stage165: positionwise input-embedding masking for low-data generalization

## Fixed hypothesis and decision rule (written before quality results)

Stage143's complete-validation BPB is 1.399686162, leaving 0.049686 to the
aspirational 1.35. Existing Stage54 training uses 0.1 elementwise embedding
dropout, 0.1 vocabulary-row dropout, and R-Drop; it does not drop a whole
input-token embedding independently at each sequence position. Its strong
training/validation gap makes memorization of particular token contexts a
plausible error source. In Stage151, 57,247 medium-frequency validation
targets absent from their current prefix carry high mean loss. Unlike
changing loss weights (Stage161), masking the representation of a *past
input* asks the model to predict using neighboring context instead of one
memorized cue. The clean input IDs remain available to the causal prefix-
copy route; only neural input embeddings are masked during training.

The broad inspiration is Wu et al., [TLM: Token-Level Masking for
Transformers](https://aclanthology.org/2023.emnlp-main.871/) (EMNLP 2023).
This pilot is **not** their attention-connection masking algorithm, and
their task results do not predict a gain here. The fixed implementation
multiplies each training input embedding by a shared-across-features
Bernoulli(0.95) mask scaled by 1/0.95, independently per batch position and
R-Drop forward. At validation/inference the mask is off, so the deployed
architecture, parameter count, FLOPs, CPU time and assets are identical to
Stage54 for a given trained state. No model sees future tokens or another
evaluation window.

Use Stage54's exact initial weights, seed 17, batch 32, sampled-window
generator, training text, primary/deep/future objectives, R-Drop, optimizer
and first 2,400 updates of its 7,200-step learning-rate schedule. The
candidate and control each see **19,660,800 primary next-token targets**.
Before training require identical initial state, exact deterministic
inference output, normalized causal/independent-row predictions, and finite
training gradients. Validate on all 376,599 targets every 300 updates,
but use only the prespecified 2,400-step endpoint for the go/no-go decision.
No rate/seed/checkpoint sweep is allowed.

The exact Stage54 seed-17, step-2,400 complete GPU FP32 control is
**1.519950369 BPB**. A full 7,200-step run is authorized only if the
Stage165 endpoint is at most **1.504950369 BPB**, a gain >=0.015. A passing
pilot would still need its predetermined late-checkpoint average, complete
CPU FP32 validation below Stage143, compatible train-only count/gate
calibration, three fresh CPU/RAM/asset resource repetitions, and a clean
extraction. Training-only regularization cannot by itself prove these.
If the quality gate fails, stop without a full run or test scoring.
Stage143 stays protected; test is untouched until a final method freeze.

The strongest objection is that 0.1 row dropout plus 0.1 elementwise
dropout may already regularize embeddings sufficiently. Whole-token masking
could also remove useful local evidence and harm the prefix-copy gate's
learned query/key alignment, even though the clean copy IDs remain.

## Observed pilot and decision

The scheduled Windows job completed 2,400 updates, presenting 19,660,800
primary next-token targets in 628.41 training seconds; full-validation
scoring took 16.23 seconds. The three new structural tests and fixed-file
checks passed before training, and fixed-file verification passed again
afterward. Its final checkpoint SHA-256 is
`51abbbccb6150275b7be10918f4b835ef1c5d5d63c2e3086629e9d407310f7fb`,
also independently verified on the Windows file. Peak CUDA allocation and
reservation were 5.648 and 5.985 GB.

The predeclared step-2,400 complete GPU FP32 validation score was
**1.535982945 BPB** on all 376,599 targets and 1,148,007 raw bytes.
It is **0.016032576 BPB worse** than the matched Stage54 control's
1.519950369, not >=0.015 better. All eight 300-step points were worse
than their corresponding Stage54 points. This rejects the *specified*
0.05 positionwise masking recipe under the matched schedule, not every
possible token-level regularizer. The task receipt reports completion
without error. Raw plans, metrics, progress and transcript are under
[`../results/stage165-evidence/`](../results/stage165-evidence/).

The quality gate failed. **Do not launch a 7,200-step run, select an
intermediate checkpoint, calibrate counts, resource-qualify or test-score
Stage165.** Stage143 remains the development candidate.
