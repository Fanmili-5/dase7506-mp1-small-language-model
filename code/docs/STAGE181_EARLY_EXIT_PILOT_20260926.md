# Stage181: intermediate-layer readout pilot

## Question and frozen decision rule

Could the existing Stage92 hybrid Transformer produce a complementary
prediction from its fourth or sixth block without running a second backbone?
This tests a **zero-training** intermediate readout only: apply the existing
final normalization, tied vocabulary matrix and output bias to each captured
block state; mix that normalized vocabulary distribution with the model's
unchanged final distribution. The readout is causal and shares all weights.
The final model's prefix-copy route remains in the final distribution; the
intermediate readout has no separately fitted copy head.

Before the pilot, fix layers 4 and 6, probability weights 0.05/0.10/0.20,
the first four independent validation batches (32,768 targets), and the
continuation gate: at least **0.02 bit per target** improvement over the same
Stage92 final predictor. A passing pilot would still require full validation,
Windows CPU/RAM/assets qualification, and comparison to Stage143 before any
promotion. This pilot is not a complete BPB or test result.

## Result

| Candidate | Delta bits/target vs Stage92 final; lower is better |
| --- | ---: |
| Layer 4 readout alone | +0.843368 |
| Layer 6 readout alone | +0.475059 |
| Layer 4, 5% mix | **-0.004093** |
| Layer 4, 10% mix | -0.000672 |
| Layer 4, 20% mix | +0.017923 |
| Layer 6, 5% mix | -0.002406 |
| Layer 6, 10% mix | -0.001710 |
| Layer 6, 20% mix | +0.005693 |

The best predeclared mix misses the 0.02-bit gate by almost fivefold. We stop
this *untrained readout* route: no full-validation sweep, inference graph
change, resource claim, or test scoring. This does not disprove a separately
trained deep-supervision head, but prior Stage54 auxiliary heads were
training-only, and the observed zero-training complementarity is too small
to justify that additional model-training commitment near the submission
deadline. Stage143 remains the protected resource-qualified candidate.

Reproduction: `scripts/diagnose_stage181_early_exit.py` with the Stage92
average checkpoint, CUDA FP32, four batches and fixed protocol. Exact
checkpoint/source/data hashes, metrics and the no-test flag are in
`../results/stage181-early-exit-pilot.json`.
