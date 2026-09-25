# Stage162: falsify sparse second-Transformer routing before engineering it

Stage143 scores 1.399686162 complete validation BPB and consumes 3.617702x
baseline CPU time with 55,810,412 bytes of inference assets. The fixed
Stage143/Stage155 50:50 mixture scores 1.376182664 but deploys two large
graphs and has not passed the CPU/64-MiB gates. Distillation (Stages157/158)
did not transfer enough gain into one graph. Before paying for quantization,
conditional routing and a new gate, calculate an **oracle upper bound** for
using Stage155 on only some independent windows.

For each 256-target validation window, compare the known Stage143 loss with
the fixed 50:50 mixture loss, using exact frozen Stage143 target log-probs and
a newly generated Stage155 target log-probability stream. The oracle selects
the most beneficial windows after seeing their validation labels. Report
complete BPB if at most 10%, 20%, 30%, 35%, 50% or 100% of windows invoke
the second expert; skip windows with negative gain. Also report the original
Stage143, Stage155 and unconditional 50:50 scores. This is a deliberately
unattainable hindsight bound, **not a valid inference gate**, leaderboard
score or checkpoint selection. An actual whole-window gate could use at most
the first input token without future leakage; even the oracle's advantage
cannot be assumed learnable from that token.

Stage155's input-only four-thread graph preflight measured about 1.885 sec
per 32x256 batch versus Stage143's 1.256 sec. Calling the second graph for
30% of windows projects roughly 26 extra seconds over Stage143's 86-second
full validation, leaving narrow CPU headroom before overhead. The combined
FP32 assets exceed 64 MiB and would require a separately verified compressed
export. Therefore the 30% oracle is the primary feasibility gate; 35% and
higher fractions are diagnostic only. Require **<=1.33 BPB** at 30% to leave
at least 0.02 BPB for the expected loss of a causal, input-only gate. If it
misses, stop this routing route; do not build or test a quantized second
graph. Passing would authorize only a train-only input-gate and resource
preflight, not a deployable score.

Verify both frozen checkpoint SHA-256 values, Stage143 cached-array hash,
376,599 complete targets, 1,148,007 raw bytes, independent known BPBs,
finite log-probs and exact unconditional 50:50 score before computing oracle
cells. Save the newly computed Stage155 target stream with its source hashes.
Only validation is read; test remains untouched. No labels or target-derived
groups may be used by a future predictor.
