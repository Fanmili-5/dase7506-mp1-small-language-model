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
