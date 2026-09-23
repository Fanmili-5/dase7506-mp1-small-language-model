# Stage53: resource-qualify the R-Drop/MKN fixed mixture

Stage50's preregistered fixed validation grid selected modified-KN count weight
0.075 with 1.4432653358 BPB, improving the Stage47 R-Drop neural average by
0.0073476674. Stage53 serializes exactly that pair of frozen experts and scalar
weight using the existing collapsed sparse recurrence. It does not use the
validation-fitted dynamic gate.

The exported graph is independently scored on CPU FP32 and measured with three
alternating baseline/candidate repetitions. It advances only if CPU time is at
most 5x, peak RSS at most 4 GiB and conservative uncompressed inference assets
at most 64 MiB. No further scalar calibration and no test scoring occur here.

## Result

Independent CPU FP32 validation reproduced **1.4432653563 BPB**, within
2.1e-8 of the Stage50 scan. Three alternating repetitions measured a median
candidate-to-baseline CPU ratio of **4.960636815x**. Peak RSS was 2,045,399,040
bytes and conservative assets were 44,208,866 bytes, so all three limits pass.

The qualified checkpoint SHA-256 is
`fb3213c2c294e7a35abb5e0716a7593c624afb9dc8514a4c4edcd3797ac58835`.
Stage53 improves the previous qualified Stage39 score by 0.0031966629 BPB and
becomes the current validation leader. No test split was scored.
