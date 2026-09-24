# Stage135: single-route successor-copy transfer to Stage115

The exact within-window cache in Stage132 crosses 1.4 BPB but adding it as a
third route costs 9.48% CPU in Stage133. Stage40's *single-route* successor
copying instead improved an older Transformer by 0.001010 BPB after training,
without increasing its attention-route count. Stage115 is only 0.000225 BPB
above 1.4, so value-alignment is worth testing as an inference-only transfer
before a more costly train-only adaptation.

Keep the exact Stage115 checkpoint, neural weights, train-derived five-order
count tables, calibrated logits and trained confidence gate. Change only the
copy route: at prediction position `t>0`, attention key at `j<t` contributes
its observed successor input token `x_(j+1)`, never `x_j`; `t=0` uses only the
vocabulary route. The key at `j>=t` is masked. No new route, parameter, data,
tokenizer, gradient, external state, or validation-fitted coefficient is added.
The causal copy helper must be tested for normalization and future-token
invariance before scoring.

The first gate is one exact exported implementation scored with the fixed
evaluator on **complete validation** in GPU FP32. Require all 376,599 targets,
normalized finite output, and BPB <1.4 before any CPU resource run. If it
fails, the untrained value swap is only a negative *transfer* test: Stage40
trained its query/key under successor semantics, and a separately registered
train-only copy-head adaptation may still be considered. If it passes, run
CPU FP32 complete validation and three fresh-process CPU/RAM/asset audits;
the old Stage115 5.170x preflight means quality alone is insufficient.
Stage85 stays the qualified fallback. Test is not scored.

## Untrained transfer result

Seven existing/new successor tests passed. The exported model was causal
(zero observed future-prefix difference) and normalized (maximum log-row-sum
error `1.92e-6`). The unchanged evaluator scored all 376,599 validation
targets in GPU FP32 at **1.559767700 BPB**, dramatically worse than
Stage115's 1.400225. This decisively rejects an *untrained value swap*.
The original content-copy query/key and gate were optimized for different
value semantics; this observation is consistent with that mismatch, but does
not establish that a trained successor head would fail. The quality gate
failed, so no CPU resource or test score was run. Raw export/source hashes
and evaluator output are in `../results/stage135-evidence/`.
