# Stage113: repair a seven-block Transformer

Stage112 identified original block 5 as the lowest-cost whole-block removal.
It is still much worse than the full teacher without training (1.442601479
versus 1.401288462 BPB). Stage113 starts from the exact Stage92 parameter
average, removes that block and physically serializes the remaining seven.
The fixed Stage92 average supplies teacher distributions on supplied training
prefixes only. The student is trained for exactly 2,400 optimizer steps,
batch 24, seed 113017, AdamW peak LR 3e-5, with 75% teacher cross-entropy
and 25% hard-label next-token NLL. The order-six count model is frozen and
used only in validation scoring; no validation or test gradient is used.

The predeclared monitoring points are 0/300/600/900/1200/1500/1800/2100/2400.
Checkpoints at 900/1200/1500/1800/2100/2400 preserve the trajectory.
The fixed validation monitor uses Stage94 neural calibration plus order-six
MKN weight 0.0625, the same as Stage112. An average of the last four saved
checkpoints is produced after training. A low score alone is insufficient:
the averaged seven-block neural must be integrated with the train-learned
gate, exactly exported, and independently measured for 5× CPU, 4 GiB RAM,
and 64 MiB assets. Preserve the qualified Stage85 fallback. Test remains
untouched until final freeze.

## Result

The warm start exactly reproduced Stage112 at 1.442601479 BPB. The fixed
trajectory improved through step 2400 to **1.422172985 BPB**; the four-late-
checkpoint average scored **1.422508959 BPB** (SHA-256
`856973b93225202a27b63373c5bd980a2e8513d668c25ff257576fed86c00893`).
The run processed 14,745,600 new training targets in 355.96 training seconds
on the RTX 3070 Ti Laptop GPU. The 0.02043-BPB repair is real but still
0.02217 above the requested threshold and much worse than Stage85. Therefore
no gated export or full CPU resource test is justified for this candidate.
The seven-block architecture is retained as a negative structural control;
Stage85 remains the qualified candidate, and the test was not scored. Raw run
metadata and full validation history are in
`code/results/stage113-evidence/`.
