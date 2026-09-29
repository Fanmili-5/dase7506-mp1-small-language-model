# Stage68: output-biased neural plus frozen MKN

Stage67 adds a train-fitted 2,048-value vocabulary intercept and improves the
Stage65 neural average from 1.4153287466 to 1.4150978608 BPB. Stage68 performs
one fixed scalar validation scan with the unchanged Stage25 train-only modified
Kneser-Ney expert. The selected candidate is serialized through a fused
probability path that adds the intercept inside the existing vocabulary softmax,
then applies the same collapsed sparse backoff recurrence.

The neural, count and baseline hashes are pinned. Validation selects only the
predeclared scalar grid (0.0000 through 0.2000 by 0.0125); it supplies no gradient
to the output bias or any other tensor. Because Stage66 has only 0.85% measured
CPU headroom, Stage68 must pass three new alternating fresh-process comparisons,
plus the unchanged RAM and asset limits. Failure retains Stage66. Test remains
untouched.

## Result

The fixed grid again selected MKN weight **0.0625**, at 1.4101617729 BPB.
Independent CPU FP32 evaluation reproduced **1.4101618031 BPB**. Three
alternating fresh-process measurements gave **4.983328582x** baseline time,
2,039,746,560-byte peak RSS and 48,561,660 conservative asset bytes. All limits
pass, but CPU headroom is only about 0.33%. The qualified checkpoint SHA-256 is
`e13364bbd4d6e9436f6371e34889c0d882662866a69584b3a921c9aa2ceddae0`.
Stage68 replaces Stage66 by just 0.0001046025 BPB; output bias and MKN therefore
appear strongly overlapping and this calibration branch is closed. No test
split was scored.
