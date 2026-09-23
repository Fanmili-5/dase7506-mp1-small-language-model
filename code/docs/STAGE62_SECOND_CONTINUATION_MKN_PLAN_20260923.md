# Stage62: fixed MKN mixture after the second continuation

Stage61 improves the frozen hybrid-conv neural average to 1.4178313584 BPB
without changing the deployed graph.  Stage62 repeats the already declared
count-weight grid from 0.0000 through 0.2000 in 0.0125 increments, pairing the
frozen Stage61 neural checkpoint with the unchanged Stage25 train-only MKN
expert.  The single best scalar is frozen before export.

The collapsed implementation, equivalence tolerance, independent CPU FP32
score, three-repeat alternating CPU benchmark, peak-RSS accounting and
conservative asset accounting are unchanged from Stage57.  No target-dependent
routing occurs and test remains untouched.

## Result

The fixed grid again selected count weight **0.075**, at 1.4125901742029008
BPB.  The collapsed checkpoint independently reproduced **1.412590201060477
BPB** on CPU FP32 with maximum smoke-test log-probability error
3.814697265625e-06.

Three alternating measurements produced a candidate-to-baseline CPU ratio of
**4.950540141x**.  Peak RSS was 2,040,070,144 bytes and conservative assets
were 48,551,655 bytes, so all limits pass, although CPU headroom is only about
one percent.  The qualified checkpoint SHA-256 is
`e509d26008ed587790091ac0cd831d82b84a074075385f54dbfe9a7af8cf20a4`.
This replaces Stage57 as the qualified validation leader by 0.0026637959 BPB;
no test split was scored.
