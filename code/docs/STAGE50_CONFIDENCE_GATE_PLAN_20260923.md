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

## Result

The fixed mixture grid selected count weight 0.075 at **1.4432653358 BPB**,
improving the Stage47 neural model by 0.0073476674. The two legal-feature gates
were fit in opposite directions on contiguous validation halves. Their combined
out-of-half score was **1.4369943013 BPB**, another 0.0062710345 below the best
fixed mixture.

The first-half-trained gate evaluated on the second half with mean count weight
0.10675; the reverse direction used 0.11346. Both learned higher count weight
for higher neural entropy and stronger count confidence, and their 10/50/90%
weight quantiles were similar. This supports a real prefix-confidence signal,
but the score remains a validation-fit diagnostic and no gate was exported.

Stage53 therefore qualifies the legal fixed 0.075 mixture immediately. A
dynamic gate may advance only after separately trained proxy-neural predictions
permit train-only calibration. Test remains untouched.
