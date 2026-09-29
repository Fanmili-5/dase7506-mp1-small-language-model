# Stage149: distant co-occurrence as a train-only topic expert

Stage143's order-six MKN and the Stage146/148 exact-history diagnostics
capture local form but did not find enough complementary signal to approach
1.35 BPB. A different hypothesis is that tokens five to 32 positions back
carry topical associations for the next token even when the precise suffix
never appeared in training. This conditional bag-of-context model is a
train-only probability distribution, not retrieval of validation text.

Prespecify distance ranges `5..16` and `5..32` (inclusive), and additive
prior strengths `100` and `1000` pseudo-observations times the train unigram.
For each range, count every `(context_token, future_token)` pair at every
listed distance in the supplied training stream. At an evaluation position,
average the resulting row-normalized token distributions for eligible context
tokens inside that independent 256-token window. Positions with no eligible
context retain Stage143 unchanged. Score fixed global mixtures at weights
`0, .02, .05, .10, .20` on all 376,599 validation targets using the
hash-checked Stage143 target-probability cache. Report all 20 cells and
training/cache hashes; never read or score test.

This is a target-only quality diagnostic, not yet a deployable predictor. A
full-vocabulary implementation would need causal normalization, asset size
and CPU qualification. Build one only if this scan gains **at least 0.010
BPB** over Stage143, because the remaining 0.049686 gap demands a much larger
mechanism than the prior lookup tests. Main risk: the Transformer already
uses topical context, so pairwise co-occurrence may add little and might
overweight frequent but uninformative tokens.

## Result and decision

The prespecified complete-validation target-only scan found **no positive
mixture** in any of its 16 nonzero cells. The least harmful tested cell,
distance `5..16`, prior strength `100`, weight `0.02`, worsened Stage143 from
**1.399686162** to **1.404737480 BPB**. The `5..32` variant was slightly
worse. All four rows, fixed hashes and train-pair counts are under
`../results/stage149-evidence/distant-cooccurrence-scan.json`.

Decision: reject this topic expert; no inference matrix, CPU qualification or
test score was produced. The result is specific to this pairwise count model
and these prespecified smoothing/weight choices, not a proof that every
contextual-topic model must fail.
