# Stage214: locate complementary errors before choosing another backbone

This is a validation-only, retrospective diagnostic, not a predictor or a
new score. The protected Stage143 predictor scores 1.399686162 BPB on full
validation; the over-budget Stage143/155 equal mixture scores 1.376182664.
We need to learn whether that complementarity is concentrated in a failure
mode that a small, deployable architecture could plausibly address. Neither
number establishes the student's <1.35 objective.

Use the two SHA-pinned, complete-validation target-log-probability arrays
from Stage162 and the unchanged supplied train/validation text and tokenizer.
No test text, test probabilities or test labels are opened. Before measuring,
fix three mutually exclusive *true-target* spelling categories: ASCII word
start (`Ġ` followed only by letters), ASCII word continuation (only letters,
no `Ġ`), and all other tokens. Cross these with train frequency 100–999 versus
other, and with whether the target ID already occurred in the current causal
256-token input prefix. Report counts, Stage143 NLL, and the equal-mixture
gain for all 12 cells and for each marginal category. Check that cells sum
to 376,599 targets and that their mixture gain reconstructs the known
0.023503498 BPB within numerical tolerance.

The groups are defined by the *true successor* and cannot be used as an
inference-time gate. If the complementary gain is mostly on ASCII word
continuations, a word-internal representation deserves a bounded new pilot;
if mostly at word starts or spread across categories, another lexical head
is a weak bet. This diagnostic can eliminate an idea, not prove that any new
architecture will reach <1.35. Protect Stage143 and do not re-score test.
