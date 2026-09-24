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
