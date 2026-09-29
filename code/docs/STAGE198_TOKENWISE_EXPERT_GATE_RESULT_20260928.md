# Stage198 result: token-conditioned expert gate does not clear the quality gate

The method and synthetic gradient/normalization tests were committed at
`69bcaea` before reading the Windows result. The complete CPU FP32 diagnostic
used the protected Stage143 checkpoint and graph, the fixed tokenizer, and
all 1,472 independent validation windows (376,599 targets; 1,148,007 raw
bytes). Its custom loader verified and read validation and tokenizer only;
it did not open test text. Zero offsets reproduced Stage143 at
**1.399686162042141 BPB**; maximum full-vocabulary zero-offset probability
error was `1.1921e-7` and maximum expert mass error was `3.5763e-6`.

The fixed one-step token-offset fit used validation answers only for a
**non-deployable cross-half diagnostic**. First-half offsets evaluated on the
second half gained `310.133846` nats; second-half offsets evaluated on the
first gained `350.128933` nats. Their combined held-out score was
**1.398856412887212 BPB**, a gain of only **0.000829749 BPB** over Stage143.
Both halves improve slightly, but the predeclared `<=1.365` advancement gate
fails by **0.033856413 BPB**. The gain is far too small to justify an
out-of-fold train-only fit, CPU inference implementation, new freeze, or
another test run. The Stage198 offsets must not be deployed, submitted, or
described as a legal trained candidate.

This rejects the fixed regularized diagonal-Fisher token-offset diagnostic,
not every conceivable token-aware model. It also suggests that the large
Stage175 hindsight oracle gap cannot be recovered simply by assigning each
candidate token a stable count preference. The protected Stage143 checkpoint
and ONNX graph hashes remain `256e0e3c...3c6da3` and
`5da1de43...e5ef4`; neither was modified.

Raw evidence: `code/results/stage198-tokenwise-gate-crossfit.json`.
