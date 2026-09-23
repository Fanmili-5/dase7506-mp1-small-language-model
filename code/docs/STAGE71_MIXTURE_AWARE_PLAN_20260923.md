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

The first launch was interrupted before its initial validation completed and
before any gradient update.  It exposed a training-throughput problem: the
generic MKN module materialized all 2,048 probabilities although NLL needs only
the observed target probability.  The replacement computes the exact same CSR
backoff probability only at each training/validation target, verified against
the complete distribution.  R-Drop KL remains on both full neural
distributions.  This is an algebraic training-time optimization; final export
and official evaluation remain unchanged.

The second launch was likewise interrupted before the initial score and any
gradient update: target-only CSR evaluation still expanded every successor row.
The final lookup precomputes sorted `(context,target)` integer keys from the
unchanged CSR buffers and uses one binary search per order and target.  Unit
tests require exact equality with the complete MKN distribution.

The third launch showed that GPU binary search remained a poor sparse workload
and was interrupted before the initial score or a gradient update.  The fourth
launch keeps the immutable MKN buffers and direct target lookup on CPU, sends
only the resulting target probabilities to CUDA, and reserves GPU memory and
compute for the dense neural expert.  The probability calculation itself is
unchanged.
