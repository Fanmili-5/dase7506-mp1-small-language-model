# Stage55: fixed MKN mixture for the hybrid-conv R-Drop model

Stage54 establishes a 1.4285940452 BPB neural average while Stage25 provides a
frozen train-only modified Kneser-Ney expert.  Stage55 evaluates a preregistered
scalar count-weight grid from 0.0000 through 0.2000 in 0.0125 increments on the
validation split.  It freezes the single best scalar before export; there is no
target-dependent routing and no test access.

The selected pair is serialized with the already resource-admitted collapsed
sparse recurrence.  A deterministic log-probability equivalence smoke test
guards the export.  Independent CPU FP32 validation must reproduce the scan,
then three alternating baseline/candidate runs enforce the 5x CPU, 4 GiB peak
RSS and 64 MiB conservative asset limits.

## Result

The fixed grid selected count weight **0.075** at 1.4232064578982955 BPB.
Independent collapsed CPU FP32 scoring reproduced **1.4232064776502589 BPB**;
the deterministic export smoke test had maximum absolute log-probability error
3.814697265625e-06.  This improves the Stage54 neural average by 0.0053875675
BPB and the previous qualified Stage53 model by 0.0200588786 BPB.

Three alternating runs measured median candidate and baseline times of
62.4032160 and 12.6503774 seconds, respectively: **4.932913385x**.  Peak RSS
was 2,040,029,184 bytes and conservative assets were 48,551,655 bytes, so all
three resource limits pass.  The qualified checkpoint SHA-256 is
`9ac5edd942f387444c8bb729b2ab3d53078063a6b09035503da5a3ebfa67700e`.
No test split was scored.
