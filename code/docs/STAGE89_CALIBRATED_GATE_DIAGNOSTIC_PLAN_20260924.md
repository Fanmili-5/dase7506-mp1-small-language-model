# Stage89: calibrated causal-gate diagnostic

Stage50 and Stage60 found repeatable cross-half gains from causal confidence
features, but their experts were much weaker than the current calibrated
Stage85 predictor. Stage89 transfers the same fixed four feature sets and
two-contiguous-half cross-fit protocol to the exact Stage71 neural expert after
Stage79 temperature, train-unigram-prior and copy-gate calibration, together
with the unchanged train-only MKN expert.

Every gate input is available before the next token: neural/count confidence,
expert agreement, window position, and sparse MKN prefix statistics. Validation
targets are used to fit each opposite-half gate, so this is only a diagnostic:
no gate or checkpoint is exported. A full diagnostic feature set remains a
non-deployable ceiling. The route advances only if the low-overhead feature set
cross-fits materially below 1.4, in which case coefficients must be learned
from a held-out slice of training text using proxy experts that never trained
on that slice. Test remains untouched.

## Result

The fixed weight reproduced Stage85 at **1.4030241401 BPB**. Cross-fit results
were 1.4029101953 for n-gram order only, 1.4025554318 for sparse prefix
statistics, **1.4015963300** for neural top-2 plus sparse prefix statistics,
and 1.4011247158 for the full diagnostic ceiling. The best deployable feature
set gains 0.00142781 BPB but remains 0.00159633 above 1.4, so the prespecified
gate for expensive train-only proxy calibration is not met. Nothing was
exported and test was not scored.
