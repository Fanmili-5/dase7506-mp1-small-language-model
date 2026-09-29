# Stage117: refit a cheaper gate on training text

Stage115's exact order-five gate almost reaches the quality target but fails
CPU at 5.170× in one fresh-process preflight. A no-margin gate could omit the
neural top-2 selection and, together with cached count-row maxima, plausibly
close that speed gap. Simply masking Stage100's margin coefficient worsened
BPB to 1.400582, so Stage117 refits the remaining three coefficients rather
than treating an ablation as an optimum.

The first 90% of supplied training text fits a temporary order-five count
expert. The 90–95% segment fits the three-feature gate, while the last 5%
checks it. The neural checkpoint is frozen; its prior exposure to these train
segments is disclosed. The full-data five-order count and Stage92 neural are
then fixed for validation. Three prespecified recipes (old full, old masked,
new refit) are screened across anchor weights 0.05/0.0625/0.075/0.09/0.11
and slope scales 0.2–0.7. Validation chooses only among these bounded scalar
settings, never trains coefficients. The masked old 0.0625/0.4 control must
reproduce Stage114. Any promising candidate still requires exact serialization
and three-repeat CPU/RAM/asset qualification. Test is untouched.

## Result

The 0.0625/0.4 old-no-margin control reproduced Stage114 exactly at
1.400581906 BPB. The new train-only three-feature refit improved its best
bounded setting to **1.400478201 BPB** (anchor 0.05, slope 0.4), but remained
above 1.4. The best full four-feature setting was **1.400187297 BPB**
(anchor 0.05, slope 0.5), still above target and still carrying top-2 CPU
cost. None of the 90 predeclared settings met the quality gate. This refit
is therefore not exported; Stage115 remains an unqualified diagnostic and
Stage85 the qualified candidate. The full training fit, selection record, and
all 90 validation rows are in `code/results/stage117-evidence/result.json`.
This run did not score test.
