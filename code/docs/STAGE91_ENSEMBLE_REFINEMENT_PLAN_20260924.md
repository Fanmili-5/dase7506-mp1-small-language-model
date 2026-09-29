# Stage91: bounded heterogeneous-ensemble refinement

Stage90 beats 1.4 by a wide margin but selects the maximum tested Stage76
weight, 0.30. Stage91 makes one prespecified boundary extension: Stage76 weight
0.25--0.70 by 0.025 and MKN weight 0.00--0.10 by 0.0125. The experts,
calibration, split and probability-mixture arithmetic remain unchanged.

This is still a validation-only, over-budget diagnostic and is never exported.
The neural-only optimum is recorded separately because it defines the simplest
two-neural teacher for train-only distillation; the count expert will be
rescanned only after a single-model student is trained. Test remains untouched.

## Result

The bounded refinement selected Stage71/Stage76/MKN weights
0.5375/0.4250/0.0375 at **1.3816233893 BPB**. The separately recorded
neural-only optimum used Stage71/Stage76 weights 0.55/0.45 and scored
**1.3832024074 BPB**. The optimum is interior, and the neural-only teacher has
0.0168 BPB of margin below the requested target without requiring dense MKN
teacher probabilities. It advances as the fixed Stage92 distillation teacher;
the ensemble itself was not exported and test was not scored.
