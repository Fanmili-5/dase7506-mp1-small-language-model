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
