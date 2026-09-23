# Stage57: fixed MKN mixture after low-LR continuation

Stage56 improves the hybrid-conv neural average to 1.4204238616 BPB without
changing the deployed graph.  Stage57 repeats Stage55's already declared fixed
count-weight grid from 0.0000 through 0.2000 in 0.0125 increments, now pairing
the frozen Stage56 neural checkpoint with the unchanged Stage25 train-only MKN
expert.  The single best scalar is frozen before export.

The collapsed implementation, equivalence tolerance, independent CPU FP32
score, three-repeat alternating CPU benchmark, RAM accounting and conservative
asset accounting are unchanged from Stage55.  No target-dependent routing and
no test scoring occur.
