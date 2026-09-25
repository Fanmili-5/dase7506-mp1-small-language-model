# Stage160 candidate: context-conditioned spelling residual on one Transformer

The most recent independent small expert fit the spare inference budget but
hurt Stage143 at both fixed mixture weights. Stage155's larger Transformer
and Stage157/158 distillation likewise did not replace Stage143. This rules
out neither all architectural changes nor all word-form sharing, but it
argues against buying another full backbone or another distillation schedule.

The remaining concrete failure is Stage151's 57,247 medium-frequency
validation targets absent from their current causal prefix, with 4.700063
mean NLL. A simple byte-composed *static* token embedding previously helped
by only 0.000777 BPB (Stage65), so repeating it is not justified. The new
hypothesis is narrower and genuinely different: a **nonlinear,
context-conditioned output residual** shares statistical strength between
tokens with related spellings while retaining Stage143's full neural/copy/
count predictor. It is a second output head, not a second sequence backbone.

The fixed supplied tokenizer has 2,048 token strings, exactly 256 symbols
in its ByteLevel representation, and maximum token-string length 12 (1,997
of 2,048 have length <=8); its SHA-256 is
`020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e`.
Represent each token as its 12-position symbol IDs. A small nonlinear
transformation of the **causal hidden state** produces a context vector;
position-specific symbol embeddings sum into one output row per token. The
resulting 2,048-way residual is added to Stage143's existing log
probabilities, followed by full-vocabulary log-softmax. Initialize the
residual to zero, so step zero must reproduce Stage143. This consumes only
the current input window; symbol IDs come from the frozen tokenizer, not
from validation/test text or true next-token labels at inference.

First implement a frozen-Stage105 GPU training prototype and a CPU inference
cost/asset preflight. Require exact step-zero Stage143 validation within
2e-5 BPB, normalized causal outputs, projected total assets <=64 MiB,
and extra CPU cost plausibly within Stage143's ~33-second headroom.
If feasible, train only the residual head on supplied train targets with one
fixed seed/schedule; compare complete validation to Stage143, with a
predeclared >=0.005-BPB gain required before any integrated CPU export.
Before training, use 32x256 synthetic causal hidden/base rows to require
zero-start maximum log-probability error <=2e-6, normalization error <=1e-5,
causal-prefix and independent-row error <=1e-6, eight-repeat four-thread
head-only median <=0.45 seconds/batch, and Stage143 measured assets plus
the head's raw FP32 parameters and a 200-kB artifact reserve <=64 MiB.
These checks establish only a feasibility screen, not the official full
predictor's CPU/RAM performance.
This is a hypothesis, not an observed improvement or a promise of <1.35.
The strongest objection is that spelling may supply little semantic
information beyond the already trained token embedding; the validation
gate is designed to falsify that cheaply. Test stays untouched.
