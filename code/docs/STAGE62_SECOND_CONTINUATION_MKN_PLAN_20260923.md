# Stage62: fixed MKN mixture after the second continuation

Stage61 improves the frozen hybrid-conv neural average to 1.4178313584 BPB
without changing the deployed graph.  Stage62 repeats the already declared
count-weight grid from 0.0000 through 0.2000 in 0.0125 increments, pairing the
frozen Stage61 neural checkpoint with the unchanged Stage25 train-only MKN
expert.  The single best scalar is frozen before export.

The collapsed implementation, equivalence tolerance, independent CPU FP32
score, three-repeat alternating CPU benchmark, peak-RSS accounting and
conservative asset accounting are unchanged from Stage57.  No target-dependent
routing occurs and test remains untouched.
