# Stage57: fixed MKN mixture after low-LR continuation

Stage56 improves the hybrid-conv neural average to 1.4204238616 BPB without
changing the deployed graph.  Stage57 repeats Stage55's already declared fixed
count-weight grid from 0.0000 through 0.2000 in 0.0125 increments, now pairing
the frozen Stage56 neural checkpoint with the unchanged Stage25 train-only MKN
expert.  The single best scalar is frozen before export.

The collapsed implementation, equivalence tolerance, independent CPU FP32
score, three-repeat alternating CPU benchmark, RAM accounting and conservative
asset accounting are unchanged from Stage55.  No target-dependent routing and
no test scoring occur.

## Result

The fixed grid again selected count weight **0.075**, at 1.4152539720019623
BPB.  The collapsed checkpoint independently reproduced **1.415253996962834
BPB** on CPU FP32 with maximum smoke-test log-probability error
3.814697265625e-06.

Three alternating measurements produced a candidate-to-baseline CPU ratio of
**4.876694073x**.  Peak RSS was 2,039,488,512 bytes and conservative assets
were 48,551,655 bytes, so all limits pass.  The qualified checkpoint SHA-256 is
`c9b3e846ff1a2901b15c3b84627b242b1dfe0e7facdc90d22e8e31dec07ae4e9`.
This is the current qualified validation leader; no test split was scored.
