# Stage90: heterogeneous ensemble ceiling

Stage89 shows that a causal confidence gate alone does not close the remaining
gap. Stage90 tests whether the balanced three-attention Stage76 architecture
contains errors sufficiently different from the calibrated four-attention
Stage71 leader to justify train-only heterogeneous distillation.

The diagnostic computes target probabilities from three frozen experts: the
exact Stage79-calibrated Stage71 neural model, the independently trained Stage76
balanced-hybrid model, and the train-only Stage25 MKN model. It exhaustively
scores the fixed rectangular grid of Stage76 weight 0.000--0.300 by 0.025 and
MKN weight 0.000--0.150 by 0.0125; Stage71 receives the remaining mass. This is
a validation-selected two-neural ensemble that exceeds the deployment budget,
so it is a ceiling only and is never exported. A distillation run advances only
if the ceiling materially beats 1.4. Test remains untouched.

## Result

The current Stage85 point reproduced at **1.4030241722 BPB**. The best grid
point used Stage71/Stage76/MKN weights 0.6625/0.3000/0.0375 and scored
**1.3830667832 BPB**, a 0.01995739 gain. The neural-only 0.70/0.30 mixture
already scored 1.3852658919. Because the optimum hits the Stage76 search bound,
one bounded refinement is justified before fixing the distillation teacher.
No ensemble was exported and test was not scored.
