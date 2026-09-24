# Stage82: untied calibrated-output qualification

Stage80 reproduces the 1.40302417 BPB calibrated predictor but misses the CPU
limit by 0.42%.  Its runtime divides every hidden vector by temperature before
the tied vocabulary projection.  Stage82 removes that operation by cloning the
output matrix, dividing the clone by the frozen temperature once at export,
and leaving the input embedding unchanged.

This is an algebraically equivalent inference transformation.  It adds one
`2048 x 288` FP32 matrix (about 2.36 MiB) but no new matmul, training target or
validation selection.  A direct Stage80/Stage82 comparison must pass before
independent CPU-FP32 scoring.  The exact exported predictor must then pass CPU
time at most 5x baseline, peak RSS at most 4 GiB and total uncompressed assets
at most 64 MiB.  Test remains untouched.

## Result

The exported predictor reproduced validation BPB `1.4030241743`, and passed
memory (`2,044,596,224` bytes peak RSS) and assets (`50,925,913` bytes).  Its CPU
ratio was `5.268523x`, so it failed the 5x limit.  The extra untied output matrix
increased memory traffic enough to outweigh removing the elementwise division;
Stage82 is therefore rejected.
