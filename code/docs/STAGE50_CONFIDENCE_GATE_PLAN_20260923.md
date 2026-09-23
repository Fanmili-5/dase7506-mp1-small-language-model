# Stage50: prefix-confidence mixture-gate diagnostic

The target-conditioned neural/count oracle reaches 1.36149 BPB, but it is not a
legal predictor. Stage23's hidden-only gate failed because the full-data neural
expert had already seen its train calibration segment and the gate learned a
near-zero count weight. Stage50 tests a different mechanism before spending
another full training budget.

For every validation prefix, the diagnostic uses only information available
before the next token: neural and modified-KN entropy, top probabilities,
margins, expert agreement and cross-probabilities, window position, highest
matched n-gram order, backoff, row size and sparse probability mass. The target
is used only by the mixture log-loss objective.

Two linear gates are fitted in opposite directions on disjoint contiguous
validation halves and evaluated on the unseen half. Their combined cross-fit
BPB is the decision statistic. Because validation labels still fit these
diagnostic gates, no gate or checkpoint is exported and no result is eligible
for submission. A material cross-fit gain would justify the expensive legal
follow-up: train a proxy neural model without a held-out slice of supplied
training text, fit the gate on that slice, then attach it to the full-data
experts and evaluate validation once.

Stage50 runs only after Stage47 finishes and uses its fixed averaged inference
checkpoint plus the unchanged Stage25 train-only modified Kneser-Ney expert.
Test data remain untouched.
