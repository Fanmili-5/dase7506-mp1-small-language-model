# Stage205: compositional pair input after Stage204 overfit (pre-outcome)

This is a fixed train/validation-only architecture test. The protected
Stage143 checkpoint, graph, frozen test record and package remain untouched.

## Mechanism selected by failure analysis

Stage204's dedicated 8,192-bucket pair table improved matched step-2,400
validation BPB by only `0.010645391` despite markedly lower sampled training
loss. Over steps 2,100--2,400, its mean training primary NLL was `3.016336`
versus Stage54's `3.249586` (difference `0.233250` nats/token). The
validation gain at step 2,400 corresponds to only about `0.0225`
nats/target after the fixed raw-byte normalization. Training dropout and
deterministic validation differ, so these are not directly comparable
absolute losses, but the matched deltas suggest the extra table memorized
training-specific token pairs more readily than it generalized.

Two-sentence pitch: the small supplied corpus may not support a separately
learned representation for thousands of hashed adjacent-token identities.
Instead, a rank-64 bilinear previous/current-token interaction shares its
factors across all pairs, supplying a causal phrase feature with about
281,000 rather than 2.36 million extra parameters.

The fixed intervention keeps Stage54's eight alternating Transformer blocks,
copy head, training losses, optimizer and tokenizer. It adds two independent
2,048-by-64 learned token-factor tables, multiplies their preceding/current
rows elementwise, multiplies by `sqrt(64)`, projects 64-to-288 without bias,
and adds `0.75` times the result to the current token embedding. At position
zero, the pair residual is exactly zero; each 256-token window resets.
Factor tables initialize with normal std `0.1` and projection with std
`0.02`. These values, rank, and scale are fixed here. Training reads only
supplied train text. No target, future, external text or cross-window state
is used at inference.

The strongest objection is that Stage54's local convolutions already learn
this interaction, so an explicit factorized product may add no independent
predictive signal. The lower parameter count may also underfit; this is not
guaranteed to outperform the hashed table. It is a final test of the
**parameter-sharing mechanism**, not a rank/scale/dropout search.

## Ordered and falsifiable gates

1. Unit-test zero-scale exact Stage54 parity, causal prefix, independent-row
   behavior, normalized outputs and first-position reset. Before any corpus
   use, check one synthetic BF16 batch-32 CUDA update with <=7 GiB allocated
   and reserved memory below the GPU total. Export a random-weight FP32
   input-only graph, require OpenVINO/eager hidden maximum error <=3e-4,
   conservative graph + unchanged Stage143 other assets +262,144 source
   reserve <=64 MiB, and candidate feature median <=1.25x the Stage143
   feature graph across eight interleaved four-thread calls. Failure stops
   before training; the screen is not a final resource qualification.
2. If admitted, run seed 17, 2,400 updates, physical/effective batch 32,
   the same train-window sampler, 19,660,800 primary target presentations,
   and Stage54's first 2,400 learning rates from its 7,200-step schedule.
   The **step-2,400 endpoint** on all validation targets is the only pilot
   selection point. A fresh full 7,200-step run is allowed only if the
   candidate beats Stage54's fixed `1.5199503686120217` BPB by at least
   **0.030**, i.e. reaches `<=1.4899503686120217`. Early checkpoints do
   not override a failing endpoint.
3. A full run would still need complete CPU FP32 validation below Stage143's
   `1.399686162042141`, then three-repeat <=5x baseline CPU time, <=4 GiB
   peak RAM and <=64 MiB counted assets. The aspirational <1.35 validation
   goal and student's <1.38 full-test minimum are not inferred from a pilot.
   A new full test would require separately committed method freeze and
   authorization. If any gate fails, do not retune the pair rank, scale,
   initialization, training duration or seed after seeing validation.

## Synthetic/input-only preflight outcome, before training

The one-off Windows preflight passed its fixed gates with no corpus split
opened: max OpenVINO/eager hidden error `3.81470e-6`, conservative projected
inference assets `57,181,111` bytes (<67,108,864), eight-interleaved
candidate/reference feature-time ratio `0.949772`, and one complete
batch-32 BF16 synthetic update at `5,553,883,136` peak allocated GPU
bytes. This admits only the fixed 2,400-step train/validation pilot. The
input-only graph is random-weight, so these numbers are not a deployable
predictor score or formal CPU/RAM qualification. Raw evidence:
[`../results/stage205-evidence/preflight.json`](../results/stage205-evidence/preflight.json).
