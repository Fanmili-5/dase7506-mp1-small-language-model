# Stage64: fixed MKN mixture after primary-emphasis continuation

Stage63 improves the frozen hybrid-conv neural average to 1.4161061665 BPB
without changing the deployed graph.  Stage64 repeats the already declared
count-weight grid from 0.0000 through 0.2000 in 0.0125 increments, pairing the
frozen Stage63 neural checkpoint with the unchanged Stage25 train-only MKN
expert.  The single best scalar is frozen before export.

The collapsed implementation, equivalence tolerance, independent CPU FP32
score, three-repeat alternating CPU benchmark, peak-RSS accounting and
conservative asset accounting are unchanged from Stage62.  No target-dependent
routing occurs and test remains untouched.

## Result

The fixed grid shifted the selected count weight from 0.075 to **0.0625**, at
1.4111047936006276 BPB.  The collapsed checkpoint independently reproduced
**1.4111048225165579 BPB** on CPU FP32 with maximum smoke-test log-probability
error 3.814697265625e-06.

Three alternating measurements produced a candidate-to-baseline CPU ratio of
**4.876403704x**.  Peak RSS was 2,040,074,240 bytes and conservative assets
were 48,551,655 bytes, so all limits pass.  The qualified checkpoint SHA-256 is
`435afaf6fd4a37664879deaf1372d09d875b4ffe63f564b4def724ae6e2af10e`.
This replaces Stage62 as the qualified validation leader by 0.0014853785 BPB;
no test split was scored.
