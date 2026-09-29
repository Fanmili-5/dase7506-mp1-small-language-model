# Stage45: compute-neutral residual-width reallocation

The failed 9x240 experiment changed depth and width together and slightly reduced
quality. Stage45 keeps the accepted depth of eight but tests a different
bottleneck: whether the 256-dimensional residual/attention stream is too narrow
relative to its SwiGLU expansion.

The candidate widens the residual stream from 256 to 288 and narrows each
SwiGLU hidden layer from 683 to 528. Per block, attention plus MLP linear weights
change from approximately 786,688 to 787,968, within 0.2%; depth, head count,
copy dimension and training-only mechanisms remain fixed. Attention's quadratic
work rises with width, so equal projection parameters do not guarantee equal CPU
time.

Stage45 therefore performs only a random-weight full hybrid preflight: the
candidate neural graph plus the fixed collapsed min-count-2 modified-Kneser-Ney
expert at weight .125, measured against the same baseline for three repetitions.
It must pass 5x CPU, 4 GiB peak RSS and 64 MiB assets before any gradient update.
Passing establishes feasibility only, not quality. No test split is scored.

## Result

The three-repeat random-weight full-hybrid preflight failed the unchanged CPU
limit: median candidate-to-baseline time was **5.080119851x**. Peak RSS was
2,051,219,456 bytes and conservative inference assets were 44,529,570 bytes, so
memory and assets passed. The graph has 6,935,617 neural parameters. This
isolates the failure to runtime: wider attention and vocabulary/copy projections
cost more even though the per-block projection parameter count was matched.
Stage45 is rejected before gradient training. No validation quality claim or
test score was produced.
