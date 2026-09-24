# Stage94: post-distillation scalar recalibration

Stage92 changes the neural distribution and reaches 1.40179706 BPB under the
old Stage79 deployment scalars. Stage94 performs one fixed local recalibration
over temperature 1.05--1.15, train-unigram prior weight 0.025--0.075, copy-gate
shift 0.0625--0.3125, and MKN weight 0.0375--0.1125. The original point
1.10/0.05/0.1875/0.075 is included exactly.

No model weight changes and no new gradient target is introduced. Validation
selects the four deployment scalars from the preregistered local grid. Only a
sub-1.4 result advances to serialization and exact resource qualification.
Test remains untouched.

## Result

The old point reproduced at 1.4017970543 BPB. The local grid selected
temperature 1.125, prior 0.0625, copy shift 0.25 and MKN weight 0.0625 at
**1.4017076878 BPB**, only 0.00008937 better. Scalar calibration is effectively
closed and no checkpoint was exported. Test was not scored.
