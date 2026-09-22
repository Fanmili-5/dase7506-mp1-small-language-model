# Stage23 plan: train-only hidden-state count gate

Status: implementation prototype and tests only; not launched while Stage22 is
training. Stage21 remains the qualified candidate.

The full target-conditioned neural/count oracle is1.3614907205, but the correct
next token is unavailable to a predictor. Highest matched n-gram order alone
only reaches a diagnostic1.4727502299. This stage tests whether the frozen
Transformer hidden state contains enough legal prefix information to predict
when the count expert is useful.

`student_dynamic_gate.py` adds one linear256-to-1 gate. Its input is only the
causal final hidden state. The output weight is bounded to[1e-4,.9999], retaining
complete positive support. Initial weight is exactly.10. Neural and count
experts are frozen during gate fitting. Initial synthetic/real checks reproduce
the fixed.10 hybrid within5.722046e-6 max logp error; normalization error is
3.129244e-7. Causality, finite gradients and nonfinite-state rejection pass.

Calibration uses only supplied training tokens:

- first90%: build a temporary count table;
- next5%: fit the257 gate parameters;
- final5%: early-stop selection;
- deployment: attach the selected gate to unchanged full-train count tables.

The temporary count expert never sees either calibration segment. The neural
expert was previously trained on the whole supplied training text, which is a
known limitation recorded in the receipt. A linear gate, weight decay, bias
anchor and patience8 bound overfitting. No validation/test data fit the gate.
After fitting, official validation is called once for development selection.

The prototype currently computes a separate dense count distribution and is
expected to resemble Stage20 timing. If and only if it improves quality, its
same arithmetic must be collapsed/directly accumulated before resource
qualification. No unchanged-code timing reruns and no test scoring.
