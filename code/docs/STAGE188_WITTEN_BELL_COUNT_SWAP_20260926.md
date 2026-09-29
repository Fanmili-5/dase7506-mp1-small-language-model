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

## Complete result and stop decision

The train-only builder verified the exact Stage143 checkpoint and fixed
train/tokenizer hashes. All five Witten–Bell tables had **identical** keys,
offsets and successor IDs to the protected MKN tables; the raw
[build record](../results/stage188-evidence/build.json) lists every order's
actual support. Maximum post-FP32 per-context mass-plus-
backoff error was `5.22e-8`. A count-only input-prefix smoke test had
`1.79e-7` maximum distribution-sum error and zero causal-prefix difference.
The challenger checkpoint SHA-256 was
`c60c1fa2590fcce1f0698c8293331d30b952a59e36c997bf891a00afbe477c00`;
projected 18-file assets were **55,810,500 bytes**, only 88 bytes above
Stage143. This projection is not a fresh whole-process resource gate.

The checkpoint transfer SHA matched on Windows. The unchanged evaluator
scored the **complete** 376,599-target, 1,148,007-byte validation split on
CPU FP32 with four threads. The [raw result](../results/stage188-evidence/validation.json)
was **1.414884001808139 BPB**, **0.015197839765998 worse** than Stage143's
1.399686162042141. The 1,472-window loss sidecar summed to its reported
NLL within `2.33e-10`; its local SHA-256 is
`4a3a9952ce99ed38cb080759da80b8b9841eae6e3c8cd0a1dde6a1d06d747186`.
The transferred JSON SHA-256 matched the Windows original:
`4967a95989d7fec0c8ef6f98e09e4aa5b3f06cdeecbe2d4406c2f1bc6e1c570c`.
Git normalizes its Windows CRLF line endings in the committed text blob;
the committed LF-content SHA-256 is
`cec10a5dbf11b837fd378d0a7217c279b4829c1fa40b49a05f48195dca8f6c14`.
No test score was computed.

The predeclared >=0.015-BPB **improvement** gate fails decisively. Stop
Stage188: no gate refit, full resource qualification, clean-extract promotion
or submission of this challenger. The protected Stage143 predictor remains
the best resource-qualified development candidate. This negative result
applies to the fixed same-support Witten–Bell swap with Stage143's frozen
gate; it does not establish that MKN is optimal among all count estimators.
