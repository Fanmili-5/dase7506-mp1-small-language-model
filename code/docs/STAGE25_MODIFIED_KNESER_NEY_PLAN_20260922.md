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
