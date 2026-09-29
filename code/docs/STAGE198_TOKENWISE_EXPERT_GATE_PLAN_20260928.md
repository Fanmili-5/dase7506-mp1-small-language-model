# Stage198: token-conditioned expert gate, diagnostic only

Stage143 assigns one count/neural mixture weight to every candidate token at
an input position. Stage175's unattainable target-aware oracle left a large
gap, while Stage176's richer **position-only** gate failed to transfer across
validation halves. A different question remains: do stable token identities
require different count weights even when the causal position is identical?
The candidate token ID is available while constructing the complete
next-token distribution; the *true* next token is not available at inference.

Before seeing the result, freeze this diagnostic procedure. Keep Stage143's
checkpoint, tokenizer, independent 256-token windows and all expert
probabilities unchanged. For each token `v`, add a learned offset `b[v]` to
the logit of the existing input-only count weight. Form
`q[v] = (1-w[v])*p_neural[v] + w[v]*p_count[v]` and normalize across all
2,048 candidates. The zero-offset distribution must reproduce Stage143.
This is a full-vocabulary normalized model, not a target-only probability
shortcut. The diagnostic loader reads and verifies only the supplied
validation text and fixed tokenizer; it does not open the test file.

Partition the 1,472 independent validation windows into fixed contiguous
halves after window 736. In each half, compute the exact score gradient and
empirical diagonal Fisher of all 2,048 offsets at the zero-offset Stage143
model. Use **one** predeclared regularized step per token:
`b = clip(gradient / (Fisher + 10), -2, 2)`. Apply the first-half offsets to
the second half and vice versa, without selecting epochs, penalties, caps or
better direction after inspection. Report both held-out NLL changes and the
combined BPB. If either held-out half fails to improve (positive NLL gain),
or combined BPB exceeds 1.365, stop.
Only a pass justifies the expensive next step: learning offsets from supplied
**training text only** with a proper out-of-fold expert protocol, followed by
complete validation and exact CPU/RAM/asset qualification. Even a passing
diagnostic is not deployable, because its offsets use validation labels.

The score gate 1.365 is a screening threshold, not a test-score forecast:
Stage143's observed test minus validation difference is ~0.016 BPB, but the
shift may change for another method. The final aim remains validation <1.35
and the student's minimum acceptable test BPB <1.38. Do not inspect test,
promote a checkpoint, or alter the protected Stage143 inference files here.

Strongest objection: a token-specific fit may memorize the frequent words in
one validation half and fail on the other, and an in-sample oracle greatly
overstates deployable gain. The fixed cross-half evaluation and explicit
train-only requirement are designed to expose that failure before spending
training or packaging time.
