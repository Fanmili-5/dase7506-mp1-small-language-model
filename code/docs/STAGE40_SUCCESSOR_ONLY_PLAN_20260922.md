# Stage40: resource-matched successor copying

The current prefix-copy head attends to previous hidden states and assigns its
mass to the token at the attended position. A language-model cache can instead
use previous hidden state `h_j` as a context key and assign mass to its observed
successor `x_(j+1)`. At prediction position `t`, Stage40 permits only `j<t`, so
every retrieved successor lies in the current causal input prefix. No state is
shared between evaluator windows or calls.

The earlier Stage15 dual-route prototype added successor copying beside content
copying and failed the random-weight CPU preflight at 6.1833x, so it was never
trained. That does not test whether successor copying is a better use of the
single copy route. Stage40 replaces content-copy with successor-copy while
retaining exactly one 64-dimensional query/key route. Parameter count and dense
tensor shapes match Stage26.

This is a matched architecture comparison: 8x256 Transformer, seed 17, AdamW
(.9,.999), batch 32, 7,200 updates, 58,982,400 primary targets, learning-rate
schedule, row dropout, deep-supervision layers/weight and +2/+3 multi-token
objective all remain fixed. The primary difference is the value alignment in
the learned copy distribution. Selection is the fixed average of updates
6,000/6,300/6,600/6,900/7,200 followed by independent CPU FP32 validation.

If successor-only improves the Stage26 neural BPB by at least .003, rebuild the
same train-only modified-Kneser-Ney mixture and calibration, then run the exact
resource gate. Otherwise reject it without post-hoc gate tuning. No test split
is scored. Substantive AI assistance includes the mechanism analysis,
implementation, tests and orchestration.
