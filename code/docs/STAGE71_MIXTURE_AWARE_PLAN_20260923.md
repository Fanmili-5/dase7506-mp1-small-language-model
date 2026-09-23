# Stage71: mixture-aware neural continuation

Every accepted neural model so far was optimized alone and combined with the
train-only modified Kneser--Ney expert only after training.  Yet the legal
diagnostics show a large neural/count complementarity gap.  A fixed scalar
mixture cannot make an independently trained neural expert specialize on the
prefixes where counts are weak.

Stage71 starts from the exact Stage67 output-biased neural average and keeps the
Stage25 MKN expert frozen.  For 3,600 updates, two stochastic neural forwards
are each combined in probability space with the same fixed count weight
0.0625 used by Stage68.  The loss is the mean next-token NLL of the two final
mixtures plus symmetric-KL R-Drop regularization between them.  Gradients flow
only into the complete neural expert; count tensors remain immutable and are
derived solely from training text.  Peak learning rate is 3e-5 with a fixed
seed 71017 and a prespecified average of updates 2400/2700/3000/3300/3600.

This changes the training objective while keeping the deployed neural+MKN graph
unchanged.  The averaged neural is rescanned once on validation with the
existing fixed MKN grid, then the exact collapsed checkpoint must pass CPU <=5x
baseline, RSS <=4 GiB and assets <=64 MiB.  Test remains untouched.

The first launch exposed a training-throughput problem: the generic MKN module
materialized all 2,048 probabilities although NLL needs only the observed
target probability.  An attempted scheduled-task stop did not terminate its
descendant Python process; it later reached step 1,500 and overlapped newer
diagnostics, so that trajectory is explicitly contaminated and excluded from
selection.  The second and third launches produced no progress checkpoint.

The replacement target scorer first removed the dense vocabulary and then the
remaining CSR-row expansion.  It precomputes sorted `(context,target)` integer
keys from the unchanged CSR buffers and uses one binary search per order and
target.  Unit tests require equality with the complete MKN distribution.  The
final launch keeps these immutable lookup buffers on CPU, sends only target
probabilities to CUDA, and reserves GPU memory and compute for the dense neural
expert.  R-Drop KL remains on both full neural distributions.  These are
algebraic training-time optimizations; final export and official evaluation
remain unchanged.

The final trajectory reproduced the Stage68 start at **1.4101617921 BPB**.  At
step 300 it reached **1.4078855396 BPB**, an initial gain of 0.0022762525.  This
is an interim validation observation, not the prespecified averaged candidate.
