# Stage124: CPU graph optimization pilot for the sub-1.4 predictor

The Stage105 predictor already reaches 1.399686 validation BPB but fails the
three-repeat CPU gate at 5.512x. A Windows CPU profile of the related Stage115
predictor attributes about 62% of one-batch self time to dense matrix
multiplication, so another scalar gate tweak is not a credible 10% speed fix.
Stage124 tests whether TorchScript freezing/inference graph optimization can
speed up the unchanged Stage92 neural feature extractor, which is the dominant
common cost in both Stage105 and Stage115. This is a source- and
checkpoint-pinned, no-training pilot; it does not change tokenizer, data,
outputs or evaluator.

Trace the neural feature extractor on a 32x256 input, compare eager, frozen,
and optimized-frozen outputs on two distinct batches, then record six warmed
CPU FP32 timings per variant. Require maximum absolute hidden-state error at
most 3e-5 and median optimized-feature speed gain of at least 12% to proceed
to exact full-predictor integration and formal qualification. The threshold is
intentionally large enough to be relevant to Stage105's roughly 10% runtime
overage. If the pilot misses it, retain the negative evidence and do not
export a TorchScript candidate. Validation inputs are used without labels;
test is not read or scored.

## Result

Both frozen and optimized TorchScript feature graphs matched eager hidden
states exactly on the two fixed 32x256 input batches. Six warmed CPU FP32
timings gave eager median 2.097834 s, frozen median 2.106655 s, and optimized
median 2.121751 s. Optimized was **1.14% slower** than eager, not 12% faster.
Stage124 therefore stops at the pilot; no checkpoint is exported and no full
resource audit or test scoring occurs. Raw timings, shapes, hashes, and
Torch version are in `../results/stage124-evidence/result.json`.
