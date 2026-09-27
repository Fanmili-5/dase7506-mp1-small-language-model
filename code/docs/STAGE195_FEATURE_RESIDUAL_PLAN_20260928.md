# Stage195: frozen contextual backbone plus per-token residual

Stage194's extra attention block failed its predeclared synthetic CPU screen
by 0.0005743 above a 1.20 candidate/base ratio. Stage195 removes that entire
attention operation, not because of a validation result (Stage194 has none),
but because the frozen Stage143 Transformer has already made each hidden state
causally contextual. A feed-forward residual on those states may learn errors
the frozen output head misses at much lower CPU cost. It will still fail if the
teacher's complementary information is absent from Stage143's states.

Fixed architecture: width-288 layer normalization, width-576 GELU MLP with
residual connection, rank-64 GELU output bottleneck, 2,048-class zero-initialized
correction, and a frozen Stage143/Stage105 base. No external data, future-token
access, cross-window state, or evaluator changes. This is a new architecture
screen, not an alpha/seed retry of Stage194.

First run the same synthetic-only CPU preflight and exact Stage143 checkpoint
check, with identical thresholds: output parity and prefix causality <=1e-5;
finite normalized probabilities; base assets + exact expert tensor bytes +
262,144-byte source reserve <=67,108,864; interleaved median CPU ratio <=1.20.
If any fail, stop before training.

If preflight passes, run a *fixed* 1,200-step/batch-8 train-only pilot, seed
195017, AdamW peak LR 3e-4, 100-step warmup and cosine decay to 10% peak;
freeze the base and optimize only the residual. Compare hard NLL against a
50:50 Stage105+Stage155 mixture distillation arm with loss .5 teacher cross
entropy + .5 hard NLL. Use identical initialization, train-token draws,
optimizer, schedule and full-validation checkpoints at 0/400/800/1200.
The advance gate is best full-validation BPB <=1.384686162 and a teacher arm
advantage of >=0.010 if claiming a teacher effect. A hard-only arm may advance
on its own if it meets the absolute gate, but no teacher claim follows.

The Stage143 checkpoint remains protected. Pilot success only licenses a
resource-qualified, replicated development candidate, not test access or a
score claim. The course fixed test split stays unopened in this experiment.
