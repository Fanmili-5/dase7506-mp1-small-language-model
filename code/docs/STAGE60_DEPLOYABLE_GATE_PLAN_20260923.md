# Stage60: resource-aware confidence-gate diagnostic

Stage50 showed a 0.00627 BPB cross-fit gain from a full prefix-confidence gate,
but several features require constructing and reducing the complete dense MKN
distribution and are unlikely to fit Stage57's tight CPU budget.  Stage60
repeats the two-contiguous-half validation diagnostic on the stronger Stage56
neural expert and compares four nested feature sets:

1. highest matched n-gram order only;
2. low-cost sparse MKN prefix statistics and window position;
3. neural top-2 confidence plus the sparse prefix statistics;
4. the complete Stage50 feature set as a non-deployable ceiling.

Every feature is target-independent and causal, but validation targets fit the
diagnostic gates.  Therefore no fitted gate is exported.  A train-only proxy
calibration run is justified only if one of the low-overhead sets materially
beats the fixed 0.075 mixture across the two held-out halves.  Test remains
untouched.

## Result

The frozen Stage56 neural plus Stage25 MKN reference reproduced the Stage57
fixed-weight result at **1.4152539720 BPB** (count weight 0.075).  Two-half
cross-fit results were:

- matched n-gram order only: **1.4150228918 BPB** (gain 0.0002310802);
- sparse prefix statistics: **1.4143860116 BPB** (gain 0.0008679604);
- neural top-2 plus sparse prefix statistics: **1.4130224129 BPB**
  (gain 0.0022315591);
- full Stage50 feature ceiling: **1.4123768722 BPB**
  (gain 0.0028770998).

The two fold fits agree on the sign and approximate magnitude of the strongest
coefficients, so the signal is not confined to one contiguous validation half.
However, no deployable subset alone closes the 0.01525 BPB gap to 1.4, and the
best subset also adds top-2 inference work to a candidate with little CPU
headroom.  Stage60 therefore exports nothing.  It is retained as evidence for
a later train-only calibrated add-on after the neural expert improves; test was
not scored.
