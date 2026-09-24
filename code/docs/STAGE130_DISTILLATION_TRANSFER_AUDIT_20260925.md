# Stage130: diagnose in-sample teacher transfer versus validation quality

Stage129's merged backbone-LoRA average worsened complete validation from
1.401708 to 1.401935 BPB despite a valid, unchanged FP32 inference graph.
This read-only audit tests whether that failure is a *transfer* gap (better
teacher matching on train text but worse validation next-token likelihood) or
merely an ineffective LoRA optimization. It uses no new gradients, fitting,
candidate selection, export, or test data.

Freeze Stage92 and Stage129 averaged neural checkpoints, the two Stage91
teacher experts, Stage25 five-order MKN, and Stage94 student calibration.
Evaluate the last 10% of supplied training tokens **as in-sample diagnostic
data** and the complete validation tokens as an independent diagnostic.
For each split, record per-target neural NLL, fixed-weight neural/MKN mixture
NLL, and cross-entropy from the fixed 0.55/0.45 neural teacher to each
student. Measure teacher next-token NLL too. Use identical independent causal
256-token windows for every model; verify full validation covers 376,599
targets and reproduces Stage92/129 fixed-mixture BPB within 2e-5.

Interpretation is deliberately bounded: lower train teacher cross-entropy
with higher validation NLL would support, but not prove, teacher-signal
overfitting; training text is *not* held out from either pretrained expert.
No model is promoted from this audit. The result chooses whether the next
architecture/training proposal should address supervision transfer or raw
capacity. Test remains untouched.

## Result

The validation controls reproduced Stage92 **1.401707689 BPB** and Stage129
**1.401935281 BPB**. On the last 10% of the same training text, scored in
fixed independent windows, Stage129's neural NLL also worsened from
2.387914 to 2.388359 nats/target and teacher-to-student cross-entropy
worsened from 2.882800 to 2.883841. Its fixed neural/MKN mixture improved
there by only 0.000017 nats/target (2.363991 to 2.363974), while validation
mixture worsened by 0.000481 nats/target. Validation teacher-to-student KL
rose from 0.064726 to 0.066222 nats/target. Stage129 improved only 650 of
1,472 validation windows and 698 of 1,412 training-suffix windows.

Thus the frozen LoRA average does **not** better match the teacher even on the
fixed-window training slice; the simple story of improved train fit but
validation overfitting is contradicted. This does not identify the cause.
One specific, testable mismatch remains: Stage129 optimized random 257-token
crops, whereas this audit and the official evaluator reset state at aligned
non-overlapping 256-token windows. A controlled aligned-window LoRA run can
test that difference without changing architecture, teacher, or target budget.
The training suffix is not a held-out set, and this observation is not a claim
about generalization. Raw results are in `../results/stage130-evidence/result.json`;
no test scoring occurred.
