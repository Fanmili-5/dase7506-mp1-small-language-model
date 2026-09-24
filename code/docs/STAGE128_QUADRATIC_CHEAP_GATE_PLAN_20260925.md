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
