# Stage134: FP32 oneDNN-packed FFN pilot

## Problem-first screen

The target-only Stage132 cache reaches 1.398979 BPB, but the complete
Stage133 implementation adds 9.48% CPU time to Stage115, which already
exceeded the 5x limit. Stage105's operator profile attributed 62.7% of
one-batch self CPU time to matrix multiplication; the first FFN projection
alone cost 686.9 ms of a 2.677-second forward. The bottleneck is therefore
dense neural computation, not count-table search.

Ideas considered under the exact FP32 CPU and deadline constraints:

| Mechanism | Decision |
| --- | --- |
| Prepack frozen SwiGLU linears with oneDNN | Pilot now: exact weights, no training, targets measured bottleneck. |
| Train a smaller seven-block student from scratch | Reserve: Stage113 repair plateaued at 1.422 after 2,400 updates. |
| Learned low-rank or channel-pruned FFN | Reserve: Stage112/119 direct screens lost substantial BPB. |
| Conditional per-token FFN skipping | Reserve: high implementation and parity risk. |
| Replace learned copy attention with exact local cache | Reserve: cache covers only 12.15% of targets; unknown quality cost. |
| Remove neural top-2 gate feature | Stage117 selected result remained above 1.4. |
| Further sparse count lookup optimization | Stage105 profile assigned only 0.1% to key search. |
| TorchScript feature graph | Stage124 exact but 1.14% slower. |
| ONNX Runtime feature graph | Stage126 exact but 1.36% slower. |
| Dynamic INT8 | Stage127 too little speed, too much feature error, and FP32-rule ambiguity. |
| More local-cache Python tuning | Cannot by itself erase Stage115's pre-existing CPU overage. |
| Override the official four-thread setting | Reject: not a comparable measurement. |

Two-sentence pitch: The high-quality Transformer spends most evaluation time
in frozen FP32 FFN matrix products and therefore cannot afford the causal
cache that crosses 1.4 BPB. Test whether packing those same matrices into
oneDNN's inference representation speeds them up enough without changing
their predictions or the official evaluator.

## Pilot contract

Use the source-pinned Stage92 neural checkpoint and the unchanged first two
32-by-256 validation **input-only** batches on the Windows i7-12700H host.
Compare eager FP32 features against variants packing only the eight SwiGLU
input projections, only the eight output projections, and all sixteen.
For each variant, require maximum feature absolute error at most `3e-5`,
then time six warmed complete feature forwards using the official four CPU
threads. Advance to Stage115/133 integration only if one variant is at least
12% faster than eager on the full feature path. A single-batch pilot is not
a resource qualification; an accepted variant would still need a portable
export, full official validation, and three-repeat CPU/RAM/asset audit.
No labels, gradients, test data, external text or non-FP32 arithmetic are
used. The strongest objection is conversion between dense and oneDNN tensor
formats at every FFN boundary; this pilot measures that overhead rather than
assuming packing helps.
