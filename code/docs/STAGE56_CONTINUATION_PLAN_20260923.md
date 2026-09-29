# Stage56: low-learning-rate continuation of the hybrid-conv R-Drop average

Stage54's validation curve is still decreasing at update 7,200, but its
prespecified schedule has reached the minimum learning rate.  Stage56 starts
from the fixed Stage54 five-checkpoint **training** average and performs one
predeclared 4,800-update continuation with a fresh AdamW optimizer, peak
learning rate 0.0002, 50-update warmup and cosine decay to 0.00002.

The architecture, R-Drop coefficient, auxiliary objectives, batch size and
all deployed arithmetic remain unchanged.  The sampled-window generator uses
seed 17 and is deterministically advanced past the original 7,200 draws before
continuing, so this is not a seed search.  The dropout RNG is reset once to the
documented seed 54017 because Stage54 did not preserve its terminal CUDA RNG.

Selection is fixed before launch: average updates 3,600, 3,900, 4,200, 4,500
and 4,800, export exactly to the Stage54 inference graph, then score validation
on CPU FP32.  MKN is reconsidered only after this neural result.  The test split
is not scored.

## Result

The continuation initially disturbed the averaged start: validation moved from
1.4285940633 at update 0 to 1.4356738432 at update 300.  It then recovered
monotonically enough to cross the start by update 2,400.  The five fixed
averaging points at updates 3,600/3,900/4,200/4,500/4,800 were
1.4226356267/1.4219111688/1.4210729401/1.4209493109/1.4206118965 BPB.

The exported average independently scored **1.4204238615755378 BPB** on CPU
FP32, improving Stage54 by **0.0081701836138646 BPB** with the identical
deployed graph.  Its checkpoint SHA-256 is
`b1cb9b8f7b8d94e9b8961f446b98bd80428e73818aceb4733e7af05fbaa2146b`.
The result advances to a fresh fixed MKN scan; no test split was scored.
