# Stage133: normalized causal cache inference and CPU feasibility

Stage132's fixed Stage43 local-cache recipe lowers Stage115's target-only
validation diagnostic from 1.400225 to 1.398979 BPB. The diagnostic computes
only the probability of the observed next token, so it is not yet an
evaluator-compatible model. Stage133 builds the complete distribution using
only input-window history and compares it against that independent diagnostic.

For each window row and position, record earlier context-to-successor pairs,
choose the longest exact context of length at most eight, and activate only
for order at least two. Where active, apply exactly 0.9 times the unchanged
Stage115 distribution plus 0.1 times the normalized local successor counts;
else return Stage115 unchanged. Cache tables reset per row. No labels, future
tokens, validation answers, or external state enter inference.

Unit tests require positive normalized distributions, agreement with the
Stage43 target-probability helper, independent-window reset, and causal
invariance to future input changes. A first-batch CPU FP32 pilot compares
Stage115 and the actual full-distribution cache implementation over six warm
repeats, plus full-output probability parity against the fixed cache formula.
The quality result is not promoted unless complete official CPU FP32
validation reproduces Stage132 and three fresh-process CPU/RAM/asset runs all
pass the original 5x/4GiB/64MiB limits. Because Stage115 already exceeds the
CPU gate, any added overhead is expected to be a serious obstacle; a failed
pilot is reported as such, not hidden. Stage85 remains the qualified fallback.
Test stays untouched.

The first Windows pilot attempt stopped before scoring because its script
imported `scripts.train_experiment` rather than the existing top-level
`train_experiment` module. Its unit tests passed, but it produced no timing
or score. The corrected runner uses a new `-v2` output directory; the first
attempt is not silently overwritten.

## CPU pilot result

The corrected Windows pilot completed. Five cache tests passed. On the first
32-by-256 validation-input batch, the actual full-distribution model exactly
matched the independently constructed full cache probabilities in log space;
its maximum log-normalization error was `2.71e-6`, and maximum target
probability difference against the Stage43 causal diagnostic was `4.77e-8`.
Six warmed CPU FP32 repeats gave medians **2.679938 s** for unchanged
Stage115 and **2.934045 s** for Stage133, a **1.094818x** time ratio.

Thus the quality signal from Stage132 is numerically realizable as a causal,
normalized predictor, but this direct implementation adds about 9.5% CPU
time to a parent that already failed at 5.170x. It fails the performance
feasibility gate. No complete official validation score, checkpoint export,
three-repeat resource qualification, or test score is claimed. Stage85
remains the last measured resource-qualified fallback. Raw pilot evidence is
in `../results/stage133-evidence/probe.json`. The 1.398979 Stage132 number
remains a target-only diagnostic, not a deployable score.
