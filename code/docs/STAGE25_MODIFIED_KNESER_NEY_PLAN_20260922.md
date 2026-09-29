# Stage25: modified Kneser-Ney complement to the Stage22 Transformer

Stage23 showed that a learned linear gate over the final Transformer state does
not recover the large target-conditioned neural/count oracle gap. This stage
changes the count estimator itself. It does not change seeds, reuse validation
answers as features, or train another neural model.

The earlier count expert is pruned absolute discounting with raw counts at every
order. Stage25 implements interpolated modified Kneser-Ney: the highest order
uses raw counts, each lower order uses distinct-left-context continuation counts,
the unigram uses distinct bigram continuations, and per-order D1/D2/D3+ values
are estimated from training counts-of-counts. Additive 0.1 unigram support keeps
all 2048 outcomes positive. Pruned mass backs off rather than being silently
renormalized away.

Two predeclared pruning levels, minimum effective count 2 and 3, test the
quality/asset trade-off. Both use order 5. Each is mixed with the fixed Stage22
neural checkpoint over weights 0, .025, .05, .075, .10, .125, .15, .20 and .25.
The grid is a validation-selected setting, while every probability statistic is
derived solely from the supplied training text. Full official validation scoring
checks the best mixture from each screen. Test remains untouched.

Advancement requires beating 1.4618048027. A winner still needs collapsed
inference equivalence and fresh exact-checkpoint CPU/RAM/asset qualification.
Failure retains the Stage22 fixed absolute-discount hybrid and establishes that
the standard continuation-count estimator did not help in this constrained BPE
setting. Count-building events and neural gradient targets are reported
separately; Stage25 adds no gradient targets.

Substantive AI assistance covers the estimator design, implementation, tests,
experiment orchestration and interpretation. The student must understand and
disclose it.

## Result

Both fixed screens completed and were independently reproduced by the official
CPU FP32 evaluator. Minimum count 2 selected weight 0.125 and scored
**1.4543903596** BPB (checkpoint SHA256
`3669552af47ef7cae72c14be15359d798ae1afbe7fbd3608f21d43414a555ad5`).
Minimum count 3 selected weight 0.10 and scored 1.4600359456. Thus min2 improves
the qualified Stage24 candidate by 0.0074145488 BPB and advances to exact
collapsed-inference/resource qualification. Its unoptimized checkpoint is
44,053,514 bytes; this is not yet a resource pass. Raw build and screen evidence
is retained under `results/stage25-evidence/`, with job logs separate. No test
split was scored and Stage24 remains the qualified fallback until Stage27.
