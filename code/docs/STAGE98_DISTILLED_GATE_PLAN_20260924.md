# Stage98: distilled-model causal gate diagnostic

Stage92 with the Stage94 scalar calibration scores 1.4017076878 validation BPB.
Stage89 found a 0.00142781 BPB cross-fit gain from causal confidence features
for the older Stage85 model. Stage98 repeats the same two-contiguous-half
diagnostic with the Stage92 average, Stage94 scalars, and the same train-only
order-5 MKN expert. The feature sets and gate fitting routine are unchanged.

This is only a ceiling estimate. The gate coefficients fit validation targets
in the opposite half and cannot be exported. If the low-overhead feature set
scores materially below 1.4, a separate calibration slice of training data
must be held out from both the neural and count-expert training to learn a
deployable gate. CPU, memory, and asset limits still require independent checks.
Test remains untouched.

## Result

The Stage94 fixed-weight reference reproduced at 1.4017076623 BPB. The same
low-overhead feature set as Stage89 cross-fitted at **1.4000162855 BPB**, a
0.00169138 gain but 0.00001629 above the target. The full diagnostic feature
set scored **1.3996168618 BPB**, but it requires dense count predictions and
its coefficients were fitted to validation labels. Neither figure is a
resource-qualified or exportable candidate. The result motivates the bounded
Stage99 order-6 count screen and a train-only gate-fitting experiment.
