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
