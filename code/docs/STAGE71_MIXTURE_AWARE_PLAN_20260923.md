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
