# Stage80: calibrated Stage71 serialization and qualification

Stage80 freezes the exact winner from the closed Stage79 scalar grid.  The
supplied-train unigram correction is materialized into the existing 2,048-value
output bias, and the copy shift is materialized into the existing gate bias.
Vocabulary temperature is reproduced by scaling only the hidden input to the
tied vocabulary projection; copy features remain unchanged.  The selected MKN
weight is stored as the ordinary collapsed-mixture scalar.

Unit tests and a direct-formula smoke comparison must establish equivalence
before export.  The resulting single checkpoint is then scored independently
on CPU FP32 and measured in three fresh processes against the unchanged
baseline.  Promotion requires CPU time at most 5x baseline, peak RSS at most
4 GiB, and all uncompressed inference assets at most 64 MiB.  Test remains
untouched.
