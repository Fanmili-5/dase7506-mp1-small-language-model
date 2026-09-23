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
