# Stage188: same-support Witten–Bell count-expert replacement

This is one bounded **train-only count-estimator** experiment, not a search
over seeds, mixture weights or validation-fitted coefficients. The retained
Stage143 checkpoint, graph and qualification record stay untouched. The
question is whether its pruned modified Kneser–Ney table allocates too much
or too little probability to rare seen successors versus backoff. A
Witten–Bell allocation on the **identical retained context/edge support**
changes only sparse `mass` and `backoff` tensors, not table lookup cost or
neural weights. This isolates a smoothing mechanism and keeps the inference
asset size approximately fixed.

## Fixed protocol before the outcome

Verify the supplied train text and tokenizer against the course manifest and
the exact Stage143 checkpoint SHA-256. For orders 2–4 use the existing
distinct-left continuation counts; order 5 uses the existing raw-count base,
and order 6 the same raw extension. Retain exactly the existing contexts,
edges and `min_count=2` support, rejecting any mismatch with Stage143's
sorted CSR keys, offsets or values. For each retained context, let `N` be
the sum of retained successor counts and `T` the number of retained successor
types. Set each seen successor mass to `count/(N+T)` and backoff to
`T/(N+T)`. Keep the original continuation unigram. Consequently the
retained masses plus backoff equal one per context; after interpolation the
full 2,048-way distribution should remain normalized. No validation/test
text or labels enter table construction.

Keep Stage143's neural model, copy head, existing **frozen** four-feature
gate coefficients and all output calibration fixed. This tests the exact
count-swap predictor, not an optimally refit gate. Build one candidate
checkpoint in ignored local output, verify every table and a small causal/
normalization smoke test, then evaluate the **complete validation split**
with the unchanged CPU FP32 four-thread scorer on Windows. Compare it to the
same-host Stage143 1.399686162 BPB control. Require at least **0.015 BPB**
complete-validation improvement before considering a separately specified
gate refit, independent three-repeat CPU/RAM/asset audit and clean-extract
qualification. Any smaller gain or regression stops this route; the
candidate cannot replace Stage143. Do not score test during development.

The strong rival is that modified Kneser–Ney already estimates continuation
probability better and the frozen gate was trained for its backoff feature;
a failed exact swap does not rule out all alternative smoothing schemes.
