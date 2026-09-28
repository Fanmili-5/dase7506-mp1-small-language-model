# Stage216: learned causal relative-distance bias in the global attention core

## Hypothesis and non-overlap with previous routes

The protected Stage143 scores 1.399686162 complete-validation BPB, still
0.049686162 above the desired <1.35. Stage214 found that the independent
larger-backbone complement is distributed across target types and positions.
The Stage54 ancestor has four global RoPE attention blocks alternating with
four local gated-convolution blocks. RoPE makes relative displacement
representable but does not give each head a learned distance preference.

Stage216 adds a **per-head, per-layer learned causal distance bias** to the
attention logits in the existing four global blocks. Distances 0–15 have
individual buckets; 16–255 use 16 fixed logarithmic buckets. All 32 bias
parameters per head begin at zero, so the initial predictor is numerically
the Stage54 control. RoPE, convolution, width/depth, heads, copy head,
training objective, optimizer, seed, tokenizer and data are unchanged. This
tests contextual routing rather than more capacity, changed block order,
local-only attention, linear memory or lexical features. The strongest
objection is that RoPE and the four local blocks already encode sufficient
distance structure; in that case the matched pilot should show little gain.

## Predeclared gates

1. Implement a causal, fully normalized log-probability model. At zero bias,
   compare candidate and Stage54 over fixed synthetic and supplied **train**
   prefixes; require maximum absolute log-probability error <=1e-5 in FP32.
   A future-token perturbation must leave every earlier prediction unchanged
   to <=3e-5. Reject any nonfinite gradients. This is correctness, not quality.
2. On Windows RTX 3070 Ti, run two synthetic batch-32 BF16 training updates
   at context 256, require no OOM and peak reserved GPU memory <8 GiB. Export
   a random-weight FP32 feature graph and run interleaved input-only CPU
   timing versus Stage143. Require candidate/Stage143 feature time <=1.30;
   project the eventual graph plus the current count/head/source assets under
   64 MiB. These screens do **not** establish official predictor eligibility.
3. Only if those pass, train once with seed 17, batch 32, 2,400 updates, the
   exact first 2,400 learning rates of Stage54's 7,200-step schedule, and
   the same train-only sampling/R-Drop/deep/future objectives. Evaluate all
   376,599 validation targets every 300 steps. At the **fixed 2,400-step
   endpoint**, require BPB <=1.489950368612022 (at least 0.030 better than
   the matched Stage54 1.519950368612022). Earlier checkpoints cannot be
   selected to bypass this gate. Do not try nearby bias buckets, seeds or
   rates after seeing validation.
4. A passing pilot only authorizes a **fresh** full 7,200-step run and its
   predeclared last-five average. Before considering replacement, require
   complete CPU FP32 validation below Stage143, then three fresh CPU/RAM and
   asset measurements within 5x/4 GiB/64 MiB. The <1.35 validation target
   remains the aim. No test split is opened until a new method is frozen.

All code, data, validation and model hashes are recorded with the run. Stage143
checkpoint, graph, test record and fallback bundle remain untouched.
