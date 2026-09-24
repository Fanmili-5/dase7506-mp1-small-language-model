# Stage128: train-fitted quadratic three-feature gate

Stage117's cheapest three-feature linear gate loses 0.00025 BPB against the
four-feature gate but avoids neural top-two margin selection. Its fitted
neural maximum log probability, highest count backoff, and highest count-row
mass have plausible interaction effects: e.g., a confident neural prediction
and confident count row need not receive an additive gate response. Stage128
tests the smallest nonlinear extension, all three squares and three pairwise
products. It does not change the neural or count experts and uses no test
data.

Fit a temporary five-order MKN from the first 90% of supplied train tokens.
Use 90–95% of train tokens to fit the nine-term quadratic logistic gate, and
95–100% to select the two predetermined ridge penalties {1e-4, 1e-3}, anchor
weights {.05, .0625, .075}, and slope scales {.25, .5, .75, 1.0}. The Stage92
neural is frozen throughout. The training/selection targets affect only gate
coefficients and its 24 prespecified small scalar choices; validation does not
fit them. Then score **one selected recipe** on the entire fixed validation
set alongside the exact 0.0625 static-mixture control. Guard coverage and the
static reference score 1.401707662. No export is allowed unless BPB <1.4.

A sub-1.4 diagnostic must still be implemented in the official scorer and
pass three-repeat CPU <=5x baseline, <=4GiB RAM, <=64MiB assets, with enough
runtime margin for host variation. A mere nonlinear-score gain does not imply
resource qualification. Preserve Stage85 regardless; test remains untouched.

## Result

The train-only 95–100% selection chose ridge `1e-4`, anchor `0.05`, and slope
`1.0`. Full validation precisely reproduced the static 0.0625-mixture control
at **1.401707662** BPB, but the selected quadratic gate scored
**1.401436994** BPB. It is 0.000959 worse than Stage117's best linear
three-feature diagnostic (1.400478201) and 0.001212 worse than Stage114's
four-feature diagnostic (1.400224946). The nonlinear gate therefore fails
its quality gate and was not exported or CPU-resource-tested; no test scoring
occurred. Both train-only selection and validation results are recorded in
`../results/stage128-evidence/result.json`.

The direction of the failure is informative but not a proved causal diagnosis:
the proxy count model was trained on 90% of train tokens, whereas validation
uses the full-data count model, and the Stage92 neural had already seen both
gate-training segments. A train-only-selected slope of 1.0 may transfer too
aggressively. We do not use validation to refit the quadratic coefficients
after seeing this result.
