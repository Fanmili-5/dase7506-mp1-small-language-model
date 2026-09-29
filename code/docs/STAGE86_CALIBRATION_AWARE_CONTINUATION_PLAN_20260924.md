# Stage86: calibration-aware mixture continuation

Stage85 makes the Stage79 calibrated predictor resource-qualified at
1.40302413 BPB, leaving 0.00302413 BPB to the development target.  Stage71 was
trained against an uncalibrated neural/count mixture; temperature, train-unigram
prior and copy-gate calibration were introduced only afterward.

Stage86 starts from the exact Stage71 five-checkpoint neural average and trains
the full neural expert through the final frozen deployment objective:
temperature 1.10, train-unigram prior weight 0.05, copy-gate shift 0.1875, and
the frozen train-only MKN expert at weight 0.075.  Count tensors and calibration
constants are immutable.  Two stochastic calibrated neural forwards retain the
same R-Drop coefficient 0.5; only neural parameters receive gradients.

The single predeclared trajectory uses seed 86017, 2,400 updates, batch 32,
BF16, AdamW, peak learning rate `1e-5`, and a uniform average of updates
1200/1500/1800/2100/2400.  This seed identifies the mechanism run and is not a
seed screen.  The fixed average is scored once through the same calibrated
mixture on validation.  Only a positive result advances to recalibration,
Stage85 fused export, and fresh resource qualification.  Test remains
untouched.

## Result

The step-0 scorer reproduced the frozen calibration at `1.4030241601` BPB.
Every trained checkpoint regressed: validation at steps 300/600/900/1200/
1500/1800/2100/2400 was `1.4040808318`, `1.4044322206`, `1.4044982927`,
`1.4046046752`, `1.4047554319`, `1.4048480133`, `1.4048658892`, and
`1.4048580159`.  The predeclared five-checkpoint average scored
`1.4047516320` BPB, 0.00172747 worse than Stage85.  The best point is the
unchanged start, so this mechanism is rejected without export or resource
qualification.  Test was not scored.
