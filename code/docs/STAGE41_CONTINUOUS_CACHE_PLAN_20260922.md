# Stage41: continuous successor-cache diagnostic

Stage40 learns a resource-matched successor-copy head from scratch. Stage41 is a
separate fast diagnostic that asks whether the already trained Stage26 hidden
states contain useful context-retrieval geometry before committing further
training. It applies cosine similarity between the current final hidden state
and strictly earlier hidden states, then assigns attention mass to each earlier
state's already observed successor token.

The cache is local to one independent 256-token input and rebuilt on every
forward call. At position `t`, only keys `j<t` and values `x_(j+1)` are exposed;
there is no cross-window state or future token access. The diagnostic evaluates
a fixed six-by-eight grid of cosine scales and mixture weights on validation.
Targets are used only to score the probability assigned by each complete fixed
expert without materializing a dense 2048-way cache tensor. They never affect
attention, values or predictor features.

This screen does not create a submission candidate and makes no CPU claim. A
material gain must be reimplemented as full normalized inference, combined with
the other experts only after an explicit ablation, and independently qualified
under the 5x/4GiB/64MiB limits. No test split is scored.
