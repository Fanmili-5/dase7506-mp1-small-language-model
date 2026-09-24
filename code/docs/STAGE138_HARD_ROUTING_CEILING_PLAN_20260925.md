# Stage138: single-expert-per-window routing ceiling

Stage91's heterogeneous *probability ensemble* reached 1.383202 BPB without
counts, but evaluates both networks and cannot meet the CPU limit. Their
individual neural predictions differ, so test whether a single-expert choice
at each independent 256-token window has sufficient theoretical headroom to
justify a cheap input-only router. This is a diagnostic, **not a deployable
model**: the oracle chooses after seeing validation labels and must never be
used in a submission or reported as a legitimate score.

Pin the exact Stage71 and Stage76 checkpoints and reproduce their Stage90
neural-only validation controls (Stage71 calibration fixed at temperature
1.10, train-unigram prior 0.05, copy-gate shift 0.1875; Stage76 uncalibrated).
On every fixed validation window, calculate each expert's full-window NLL,
then choose the smaller as an explicitly label-using oracle. Also report the
constant primary and alternate scores, oracle assignment fraction and
per-window gain distribution. No count expert, third model, cross-window
state, new training, or test scoring is used.

Advance to fitting an *input-only* router on supplied training text only if
the hindsight oracle scores below 1.38 BPB, leaving at least 0.02 BPB of
headroom to beat 1.4 after routing mistakes. Otherwise stop: hard window
routing cannot plausibly recover the ensemble advantage. Even if it passes,
two stored experts must be shown to fit 64MiB with inference code, and the
official four-thread CPU/RSS gates must be tested after a real router exists.
