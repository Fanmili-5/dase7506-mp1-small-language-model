# Stage66: byte-composed neural plus frozen MKN

Stage65 improves the neural average from 1.4161061665 to 1.4153287466 BPB while
exporting the unchanged hybrid-conv inference graph. Stage66 therefore performs
the same fixed scalar validation grid against the unchanged Stage25 train-only
modified Kneser-Ney expert, serializes the selected collapsed mixture, and runs
the complete CPU FP32/RAM/asset qualification.

The scan uses weights 0.0000 through 0.2000 in increments of 0.0125. The neural,
count and baseline checkpoint hashes are pinned before the scan. No count table,
neural tensor or calibration parameter is fitted to validation, and test remains
untouched. This stage can replace Stage64 only if the exact serialized candidate
passes CPU <= 5x baseline, peak RSS <= 4 GiB and conservative assets <= 64 MiB.

## Result

The fixed grid selected MKN weight **0.0625** at 1.4102663744 BPB. The collapsed
checkpoint independently reproduced **1.4102664055 BPB** on CPU FP32. Three
alternating fresh-process resource repetitions measured **4.957418247x** baseline
time, 2,041,036,800-byte peak RSS and 48,551,847 conservative asset bytes; all
limits pass, although the CPU margin is only about 0.85%. The qualified
checkpoint SHA-256 is
`2193091a69937142ac6b01a5cd1d08b8850d97be77db8b695b65e8992488b283`.
Stage66 replaces Stage64 as the qualified validation leader by 0.0008384170 BPB.
No test split was scored.
