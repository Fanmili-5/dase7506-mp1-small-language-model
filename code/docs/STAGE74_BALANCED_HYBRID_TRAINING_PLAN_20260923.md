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
