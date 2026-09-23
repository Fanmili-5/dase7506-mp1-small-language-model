# Stage75: balanced-hybrid MKN qualification

Stage74 beats the matched Stage54 architecture control by 0.0033153 BPB with a
different resource-admitted global/local layer allocation.  Stage75 keeps the
exact Stage74 average and frozen train-only Stage25 modified Kneser--Ney expert,
scans the existing fixed count-weight grid 0.0000--0.2000 in increments of
0.0125 on validation, serializes the selected collapsed predictor, and performs
independent CPU-FP32 scoring plus three-repeat CPU/RAM/asset qualification.

No gradient training, new statistics, grid expansion or test scoring occurs.
The branch advances to low-learning-rate continuation only if the collapsed
predictor improves the Stage74 neural average and passes CPU <=5x baseline,
RSS <=4 GiB and uncompressed inference assets <=64 MiB.

## Result

The fixed grid selected count weight **0.075** at 1.4199515687 BPB, and the
collapsed checkpoint independently reproduced **1.4199515889 BPB** on CPU
FP32.  Its SHA-256 is
`38576dc84a499424438b8320614eca84aeb2e579051eabadf7d3cce788f8bc7c`.
Three fresh Windows repetitions measured a 4.793800753x median-time ratio,
2,037,317,632-byte peak RSS, and 48,629,424 conservative inference-asset bytes.
All course limits pass.  Stage75 preserves Stage74's matched architecture gain
through the count mixture and admits Stage76 continuation, but remains behind
the Stage71 leader.  Test was not scored.
