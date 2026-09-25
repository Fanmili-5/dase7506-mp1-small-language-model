# Stage174: training-only byte-boundary supervision

## Why this direction

Stage143's complete-validation score is 1.399686162 BPB, leaving 0.049686
to the desired <1.35. Stage146 located high error on next tokens absent from
the current causal prefix. Larger and deeper single backbones, exact/distant
retrieval, shared branches, morphology at inference, frequency-weighted hard
labels, token masking, SAM and teacher continuation have failed their fixed
gates. The resource limits make another inference-time expert unattractive.

The new problem-first hypothesis is that the fixed 2,048-way token label
provides weak parameter sharing across related subword strings during
training. Add a **train-only** objective that predicts the first and last
ByteLevel symbol of the true next token from the same causal hidden state.
Both symbols are deterministic functions of the supplied fixed tokenizer and
train target; no extra text, changed tokenizer, future input, or cross-window
state is introduced. At evaluation, the auxiliary heads are discarded and
the Stage54/Stage143-family inference architecture is unchanged. This is a
generic multitask-supervision adaptation, not a new algorithm claim.

The strongest objection is that next-token cross-entropy already supervises
the same information and the extra heads may worsen generalization, as the
Stage160 inference-time spelling residual did. An exact-initialization and
matched-target control plus a fixed full-validation endpoint make that
objection testable before paying for a long run.

## Considered mechanisms and evidence filter

| Mechanism | Screening reason |
| --- | --- |
| Larger single backbone | Stage155 full run still 1.409877 |
| Six-attention allocation | Stage145 early margin only 0.004699 |
| Dilated local mixing | Stage164 early regression 0.015840 |
| Shared upper branches | Stage153 resource and Stage154 quality gates failed |
| Second compact neural expert | Stage159 mixture worsened |
| Teacher-mixture distillation | Stage169 gain only 0.000246 |
| Exact long-suffix lookup | Stage146 diagnostic gain only 0.003037 |
| Hidden-state kNN | Stage163 nonzero mixtures worsened |
| Distant co-occurrence | Stage149 nonzero mixtures worsened |
| Frequency-weighted hard labels | Stage161 matched pilot worsened |
| Positionwise embedding dropout | Stage165 matched pilot worsened |
| Sharpness-aware optimization | Stage168 matched pilot worsened |
| Contextual spelling residual | Stage160 endpoint worsened |
| Training-only byte-boundary supervision | Untested, no deployed parameters; select for one bounded pilot |

## Frozen checks and decision gates (before outcomes)

1. Derive a 2,048-row `(first_byte, last_byte)` table from the **unchanged**
   tokenizer via the existing ByteLevel inverse alphabet. Require exactly one
   first and one last symbol per token. Add two 288-to-256 training-only
   linear classifiers after the final hidden-state norm. The two byte losses
   are averaged, then weighted **0.2** and added to the existing Stage54
   R-Drop/deep/future objective. Do not tune that weight on validation.
2. With seed 17, the Stage174 model's shared state and its step-zero eval
   output must exactly equal the Stage54 control, before any optimization.
   On synthetic two-row windows, verify finite loss and gradients, nonzero
   auxiliary-head and backbone gradients, causal/row independence, and
   normalized output. Verify the export removes both training-only heads and
   yields exact FP32 evaluation output.
3. Only after these checks, run one seed-17 **2,400-update**, effective batch
   32, 256-token-window pilot with the same sampled-window sequence and first
   2,400 learning rates of Stage54's 7,200-update schedule. Monitor complete
   GPU FP32 validation every 300 updates. The predeclared endpoint is step
   2,400; compare its 376,599-target BPB with Stage54's identical-budget
   1.519950369 control. Continue to a separately planned full 7,200-step
   run **only if the endpoint improves by at least 0.020 BPB**. This is a
   continuation gate, not a claim of eventual <1.35.
4. Any longer run would use the fixed last-five checkpoint average and must
   beat the protected Stage143 1.399686162 on complete CPU FP32 validation,
   then pass three alternating fresh-process baseline/candidate resource
   repetitions (CPU <=5x, RSS <=4 GiB, uncompressed assets <=64 MiB),
   causality/normalization, clean-extract reproduction and report disclosure.
   No test scoring before a later explicit method freeze.

This is a one-candidate, one-weight pilot, not a grid or seed search. If the
pilot fails, preserve the negative result and do not extend it. The Stage143
checkpoint/graph, fixed data, tokenizer and evaluator remain untouched.

## Completed pilot and decision

The Windows RTX 3070 Ti scheduled task completed normally (task result 0),
with all 2,400 updates and **19,660,800** primary target presentations.
Training took **643.60 s** excluding complete-validation calls; peak CUDA
allocated/reserved memory was **5.671/5.989 GB**. The exact remote checkpoint
SHA-256 matched the metrics record:
`f2f2b19759410cec8a47854fb59129ca34ced68e66db51ae56cce22ccd6e7a8f`.
All 17 recorded source hashes match the tracked local files; fixed course
files passed verification again after the run. The checkpoint remains on the
Windows host and is not an inference/submission asset.

At the predeclared step 2,400, complete GPU FP32 validation over **376,599
targets / 1,148,007 bytes** was **1.5200191887 BPB**. The same-budget Stage54
control was **1.5199503686 BPB**: the new loss was worse by **0.0000688201
BPB**, rather than better by the required 0.020. All eight prespecified
300-step validation points were slightly worse than their Stage54 controls.
This is a bounded negative result for the chosen first/last-byte objective
and fixed 0.2 weight, not proof that all character-aware supervision fails.

The continuation gate **fails**. Do not run a 7,200-step extension, compact
export, CPU/RAM/asset qualification or test scoring for Stage174. Stage143
remains unchanged. The [run, metrics, progress and completion receipt](../results/stage174-evidence/)
are tracked; the raw PowerShell transcript, which includes the local Windows
account name, remains on the private training host instead of the repository.
