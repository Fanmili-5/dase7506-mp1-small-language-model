# Stage74: admitted balanced-hybrid R-Drop training

Stage72 establishes that the resource-adjusted 8x300 graph fits the complete
neural-plus-MKN deployment budget.  Stage74 now runs the same 7,200 updates,
batch size 32, seed 17, BF16 training, AdamW schedule, R-Drop coefficient 0.5,
deep supervision and future-token auxiliary recipe used by Stage54.  The only
intended experimental change is the admitted global/local allocation: attention
in layers 1/4/7, gated causal convolution in 2/3/5/6/8, width 300, six heads and
SwiGLU hidden width 675.

Selection is fixed before launch: uniformly average updates
6000/6300/6600/6900/7200, export the identical inference-only graph, and score
the average once on validation CPU FP32.  Stage54's matched score
1.4285940452 BPB is the architecture control.  A result that does not beat it
ends this branch; a better result advances to the existing frozen-MKN scan and
resource qualification.  No seed search and no test scoring are permitted.

## Result

The validation curve remained below Stage54 at every matched 300-update point.
The five prespecified checkpoints at updates 6000/6300/6600/6900/7200 scored
1.4340939754/1.4323974681/1.4306599856/1.4288262042/1.4274893572 BPB.
Their exported parameter average independently scored **1.4252787868 BPB** on
CPU FP32, with checkpoint SHA-256
`52476c0a518cf81527faa3ede5d8eec09e3101094d10d7f3629a593a780c2042`.
This improves the matched Stage54 average by **0.0033152584 BPB**, so the
balanced global/local allocation passes its quality gate and advances to the
unchanged MKN scan in Stage75.  It does not yet replace the qualified Stage71
leader.  Test was not scored.
