# Stage65 mechanism ranking and validation-only plan

## Decision pitch

Stage65 adds byte-derived sharing to the current hybrid-conv leader during
training, then folds it into the existing tied vocabulary matrix for deployment.
It targets sparse BPE rows without spending any inference CPU, RAM, or asset
budget, and it starts exactly from the frozen Stage63 predictor.

## Ranked candidates

1. **Training-only byte composition on Stage63 (Stage65, selected).** Stage42
   improved its matched neural control by 0.0023376 BPB. Prediction: the same
   mechanism should improve the stronger Stage63 average by at least 0.001 BPB;
   falsifier: the fixed five-checkpoint average fails to beat Stage63.
2. **Higher-order or min-count-1 MKN.** It may complement the neural model, but
   it directly consumes the remaining asset and CPU margin. Run only after a
   better neural checkpoint exists, with the unchanged qualification scan.
3. **Train-only proxy calibration for a sparse neural/count gate.** Historical
   cross-fit evidence suggests roughly 0.0022 BPB possible gain, but the current
   CPU ratio leaves little deployment headroom. Keep as a follow-up requiring
   train-only calibration and an explicit resource gate.
4. **Inference-kernel optimization followed by architecture reallocation.** This
   has larger possible upside but a wider implementation and parity surface;
   pursue only if zero-overhead representation changes plateau.

Pure seed changes, hidden-only gates, local caches, and local-heavy layer swaps
are not selected: existing validation evidence is neutral or too small relative
to their cost. They remain controls rather than the search boundary.

## Frozen protocol and advancement rule

- Development split: validation only; do not score test.
- Starting point: exact Stage63 five-checkpoint training average, checksum-pinned.
- Training: 3,600 continuation steps, fixed seed/sampler continuation, R-Drop,
  primary-emphasis auxiliary weight 0.05, base peak LR 3e-5, fresh byte
  projection peak LR 3e-4.
- Selection: fixed uniform average of steps 2400, 2700, 3000, 3300, and 3600.
- Export: materialize byte contribution into token/head weights; deployed graph
  and parameter count must equal `student_hybrid_conv_structured`.
- Advance to an MKN scan only if the independent CPU-FP32 validation score beats
  Stage63. A gain below 0.001 BPB is retained as evidence but treated as weak.
- Final candidate must still pass CPU <= 5x baseline, RSS <= 4 GiB, and total
  uncompressed inference assets <= 64 MiB.

## Result

The zero-initialized model reproduced Stage63 at step 0 with 1.4161061871 BPB.
After an early restart disturbance, its fixed averaging points at steps
2400/2700/3000/3300/3600 scored
1.4163842487/1.4156552407/1.4155590939/1.4153863778/1.4152772081 BPB.
The materialized average independently scored **1.4153287466 BPB** on CPU FP32
(checkpoint SHA-256
`f9c8b41efea91d5594b25760c91652edc071a161983b12b22cc738988a7a9bdc`).
This improves the Stage63 neural average by **0.0007774199 BPB** with no deployed
parameter or graph increment. It is a real positive result, but it misses the
predeclared 0.001 strong-evidence threshold, so byte composition is retained as
a weak incremental mechanism rather than the next main search direction. No
test split was scored.
