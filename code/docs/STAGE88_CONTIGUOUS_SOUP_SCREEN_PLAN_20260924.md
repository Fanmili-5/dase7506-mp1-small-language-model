# Stage88: preregistered contiguous checkpoint soups

Stage87 rules out useful movement along the broad Stage67-to-Stage71 weight
direction. Stage88 asks a narrower structural question: does the fixed average
of all five equally spaced late Stage71 checkpoints include an early or late
endpoint that hurts the calibrated deployment objective?

The complete search space is fixed before scoring: all 15 contiguous intervals
among steps 2400, 2700, 3000, 3300, and 3600, including the five individual
checkpoints. Each interval is averaged in FP64 and cast back to the original
FP32 tensors, then scored through the exact frozen Stage79 calibration and MKN
expert on validation. No seed is changed, no architecture or runtime path is
changed, and no non-contiguous subset is searched. Screened weights are not
exported unless they materially beat Stage85. Test remains untouched.

## Result

The 15-candidate screen improved monotonically as early checkpoints were
removed. The single step-3600 checkpoint was best at **1.4029408484 BPB**, only
0.0000832848 below Stage85. This is real but too small to justify replacing and
requalifying the leader, and it leaves 0.00294085 BPB to the requested target.
No screened weights were exported and test was not scored.
