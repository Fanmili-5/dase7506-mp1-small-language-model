# Stage148: train-derived skip-context successor diagnostic

Stage143's contiguous order-six MKN and Stage146's longer exact suffix
diagnostic leave no evidence that simply extending adjacent history can close
the remaining 0.049686 BPB gap to 1.35. A distinct, cheap mechanism is
whether nonadjacent recent tokens carry predictive information beyond the
neural/count mixture, while matching substantially more validation contexts
than six-token exact suffixes.

Before measuring, fix the five history patterns as lags from the current
input token: `(0,2)`, `(0,2,4)`, `(0,1,3)`, `(0,1,4)`, `(0,1,2,4)`.
For each, form an **unsmoothed train-only** exact successor distribution.
Respect independent 256-token validation windows: never query a lag reaching
before the current window's first input. Combine this probability with the
cached exact Stage143 target probability at fixed weights
`0, .01, .02, .05, .10, .20`. Measure all pattern/weight cells over the
complete validation target stream, retaining the unchanged Stage143
probability when no training match exists. The script must check fixed data,
tokenizer, checkpoint and target-cache hashes and must not load or score test.

This is deliberately a **target-only diagnostic**: validation labels may be
used to read out likelihood, never as training entries, inference lookup keys
or a deployable checkpoint. Require a gain of at least **0.010 BPB** before
building any causal, normalized full-vocabulary predictor and measuring its
CPU/RAM/assets. Even a positive result below that gate is unlikely to close
a useful fraction of the 1.35 gap. The strongest objection is that skipped
patterns mostly duplicate information already learned by Stage143; this
diagnostic falsifies them at low cost.

## Result and decision

The hash-checked scan completed all five patterns and six weights on the
complete validation target stream without reading test. Its best cell was
lags `(0,1,2,4)` at weight `0.05`: **1.396527257 BPB** versus the exact
Stage143 cache at **1.399686162 BPB**, a **0.003158905-BPB** target-only gain.
That is less than one third of the predeclared 0.010-BPB advancement gate and
only about 6.4% of the gap to 1.35. All cells, training-match coverage and
file/source hashes are in `../results/stage148-evidence/skip-context-scan.json`.

Decision: do not build a skip-context index or deploy this target-only
calculation. No resource qualification or test score exists for Stage148.
