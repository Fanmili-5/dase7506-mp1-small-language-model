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
