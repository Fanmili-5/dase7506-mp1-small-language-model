# Stage196: attention residual under the actual course resource contract

This is a prospective, explicitly revised feasibility question. Stage194's
attention residual missed an **extra conservative** candidate/base CPU screen
of 1.20 by 0.0005743 and was stopped without training. That screen is not the
course's 5x-baseline rule. The measured Stage143 3.6177x baseline multiplied
by the synthetic Stage194 ratio 1.200574 gives a rough 4.34x projection,
below 5x; it is *not* a formal resource qualification. Stage195 removed the
attention and passed the synthetic screen but improved full-validation BPB by
only 0.00119. Stage196 therefore tests whether the additional causal
attention, absent from Stage195, provides meaningful quality under the
actual course budget. We retain and report Stage194's failed screen; this
plan does not retroactively change that result or use test feedback.

Before any Stage196 validation result is seen, fix the Stage194 architecture
unchanged: frozen Stage105/143 core, one width-288 eight-head causal attention
block, width-576 MLP, rank-64 zero-initialized output correction. Reuse its
synthetic checkpoint/parity/causality/asset evidence and do **not** assert
resource qualification yet. The fixed pilot matches Stage195: seed 196017,
1,200 steps, batch 8, AdamW peak LR 3e-4, 100-step warmup/cosine decay to
10%, weight decay .01, FP32 complete-validation at 0/400/800/1200. Run two
arms from identical initialization and train-token draws: 50:50 Stage105+
Stage155 teacher with .5 distillation/.5 hard NLL, and hard-only NLL. Train
only the residual; teachers and base are frozen. No test data access.

Advance only if best complete-validation BPB <=1.384686162; require >=0.010
teacher advantage over hard-only before claiming teacher benefit. If quality
passes, run *formal* three-repeat complete CPU FP32 validation timing against
the course baseline and measure peak RAM and exact uncompressed inference
assets. Reject if >5x baseline, >4 GiB, or >64 MiB. Do not run a test split
or replace Stage143 based on a pilot alone. If quality fails, stop without
resource deployment work. This is a bounded architecture test, not a seed or
threshold sweep.
