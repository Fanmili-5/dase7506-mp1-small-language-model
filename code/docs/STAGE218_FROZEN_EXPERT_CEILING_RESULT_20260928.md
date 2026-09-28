# Stage218: three-family frozen-expert fixed mixtures still miss 1.35

The fixed validation-only diagnostic in the
[pre-outcome plan](STAGE218_FROZEN_EXPERT_CEILING_PLAN_20260928.md) asked whether
compressing the already-trained Stage143, Stage155 and Stage91 predictor
families could plausibly solve the student's <1.35 validation objective.
It did not train a new model, alter the evaluator or open test data. The
true-target arrays are **label-aware analysis artifacts**, not inference
inputs; the combined predictors also exceed the course's CPU/asset limits.

The Windows cache script re-created frozen Stage91 at
**1.381623389312373 BPB** over all **376,599 targets / 1,148,007 bytes**
using its previously fixed Stage71/Stage76/MKN weights
`0.5375/0.4250/0.0375`. The three component checkpoint SHA-256 values
matched their prior evidence, and all recorded source, tokenizer and
train/validation-data hashes matched the Mac repository. The Stage143 and
Stage155 cached target streams independently reproduced
**1.399686162042141** and **1.409877270418351 BPB**; their 50:50
mixture reproduced **1.376182664188505 BPB**.

| Predeclared cell | Stage143 / Stage155 / Stage91 weights | Validation BPB |
| --- | --- | ---: |
| A | 0.50 / 0.25 / 0.25 | 1.373750318 |
| B | 1/3 / 1/3 / 1/3 | **1.371030790** |
| C | 0.25 / 0.25 / 0.50 | 1.371062293 |
| D | 0.50 / 0 / 0.50 | 1.384086136 |
| E | 0 / 0.50 / 0.50 | 1.372090132 |

Cell B was also reproduced independently with nested `logaddexp` arithmetic,
giving exactly `1.371030790443056`. It is **0.021030790 above** the desired
1.35 validation target, even before the substantial challenge of fitting
all predictor assets and CPU work into the 64-MiB/5x limits. This is a
*fixed-cell screen*, not a mathematical optimum over all possible weights
or architectures, and it says nothing verified about a fresh test score.
No further validation-weight grid is run to chase a tiny diagnostic gain.

**Decision:** do not spend the remaining deadline trying to compress these
specific frozen experts as the main path to <1.35. Compression would address
their resource failure but not establish the missing quality gain. A new
predictor mechanism or better training generalization is still needed.
Stage143 remains the only protected resource-qualified candidate, at
1.399686162 complete validation and 1.415657617 already-frozen complete
test, so the coursework objective remains unfinished.

Evidence: [Stage91 cached target probabilities](../results/stage218-evidence/stage218-stage91-target-logp.npy),
[cache provenance](../results/stage218-evidence/stage218-stage91-target-logp.json),
and [fixed-mixture computation](../results/stage218-evidence/fixed-mixture-ceiling.json).
