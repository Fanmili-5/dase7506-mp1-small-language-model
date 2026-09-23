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

## Result

The materialized predictor independently reproduced the closed Stage79 result
at **1.4030241744 CPU-FP32 validation BPB**.  Its checkpoint SHA-256 is
`23e458f0870d25ed0487d9c72082cc13045e0d70ce2621eacb35f4fd789d9373`.
The direct-formula smoke comparison had maximum probability error
1.37e-6, and all unit tests passed.

RAM (**2,040,823,808 bytes**) and conservative inference assets
(**48,565,480 bytes**) pass, but the three-repeat CPU ratio is
**5.020862881x**, exceeding the hard 5x limit by about 0.42%.  Stage80 is
therefore a quality-positive but unqualified result and cannot replace
Stage71.  The likely overhead is the per-forward hidden division used to keep
the tied input/output matrix unchanged.  Stage82 will materialize a separate
temperature-scaled output matrix, trading about 2.36 MiB of asset headroom for
removal of that runtime operation while preserving predictions.  Raw evidence
is in `results/stage80-evidence/`; test was not scored.
