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
