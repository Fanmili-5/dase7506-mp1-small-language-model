# Stage33: deeper/narrower Transformer preflight

The current 8x256 inference graph is close to the 5x CPU limit, so adding a
ninth layer directly is not responsible. Stage33 tests one fixed depth/width
reallocation: 10 layers, width 224, seven 32-dimensional heads. SwiGLU ratio,
RoPE, RMSNorm, prefix-copy-64 and all inference semantics remain unchanged.
The backbone has fewer matrix parameters than 8x256 while offering two extra
nonlinear attention/FFN stages. This is an architectural hypothesis, not a seed
or learning-rate search.

Before any gradient training, instantiate the exact inference graph with random
seed-17 weights, attach the fixed minimum-count-2 modified-KN tables at weight
0.125, serialize it through the already-qualified collapsed recurrence, and run
three fresh-process CPU FP32 comparisons against the unchanged baseline. Timing
is weight-value independent for this dense graph. Reject before training if the
median ratio exceeds 5x, peak RSS exceeds 4 GiB, or conservative inference
assets exceed 64 MiB.

On a pass, a later fixed run will use the same 7,200 updates, batch 32, seed 17,
optimizer and 58,982,400 primary targets as Stage26, with intermediate heads at
layers 5/8 and the same +2/+3 training-only prediction heads. This makes the
quality comparison a matched-target depth/width reallocation. Validation only;
no test scoring or automatic architecture grid.
