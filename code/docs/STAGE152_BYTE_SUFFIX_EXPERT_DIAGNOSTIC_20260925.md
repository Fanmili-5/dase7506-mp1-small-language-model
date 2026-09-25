# Stage152: train-derived byte-suffix successor expert

The Stage151 residual audit found 57,247 validation targets with training
frequency 100–999 that are unseen in their current 256-token input prefix;
their average NLL is 4.700063 nats. A fixed exact token-suffix lookup added
only about 0.003 BPB, so the next cheap question is whether **sharing
contexts by their final raw bytes**, irrespective of BPE segmentation, gives
better coverage for spelling-like continuations. This differs from Stage42/65
byte-composed embeddings: it is an explicit train-derived conditional
probability expert rather than a training-only embedding feature.

Before results, fix context lengths `2, 3, 4` bytes, additive unigram-prior
strengths `1, 10` and mixture weights `0, .02, .05, .10, .20`. For every
training-token boundary, count the next BPE token conditioned on the last
`k` bytes of the observed training prefix. At validation, construct keys only
from bytes of input tokens inside the current independent 256-token window;
never include target bytes or earlier windows. For a matched context, smooth
its 2,048-way successor distribution with the supplied-training unigram.
For unmatched contexts, retain Stage143 unchanged. Use the hash-checked
Stage143 target log-probability array only for retrospective scoring of all
fixed cells on validation. The raw-byte reconstruction from the fixed BPE
token stream must exactly match both supplied train and validation text.

This target-only diagnostic is not a submission predictor; a promising cell
would still need a causal full-vocabulary implementation, compact train-only
table, and exact CPU/RAM/asset audit. Require at least **0.010 BPB** gain
before any such work. Strongest objection: the neural backbone and MKN may
already encode the same short spelling context, making this only another
milliscale expert. No test split is read or scored.
