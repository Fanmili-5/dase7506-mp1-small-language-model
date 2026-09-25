# Stage176: richer causal-feature gate feasibility (diagnostic only)

Stage175 found an unattainable 1.300186476-BPB per-target oracle between the
unchanged Stage143 neural and order-six MKN experts. The deployable Stage143
gate scores 1.399686162. To reach the 1.35 goal with **fixed experts**, a
new input-only gate would need to recover about half of the hindsight gap.
Earlier low-dimensional gate cross-fits gained only about 0.001–0.002 BPB,
and a train-only proxy fit suffered calibration mismatch. Before training
proxy experts, ask whether richer *causal* features can even approach 1.35.

## Frozen diagnostic before the result

Run the exact Stage143 checkpoint and supplied evaluator window generator on
all 376,599 validation targets, FP32, CPU, four threads. Do not change
model weights, tables, graph, tokenizer or scorer. At each target, recover
the fixed neural and count expert target probabilities as in Stage175. Only
the gate inputs use current-prefix information: the existing neural hidden
state, existing gate weight, neural top-two confidence, within-window
position, and MKN highest-match backoff/max-mass indicators. No target ID,
target frequency category, future token, cross-window state or cached answer
enters these features. Capture the original model's target log probability
and require exact Stage143 full-validation reproduction before gate fitting.

Partition **independent 256-token validation windows** into contiguous halves
without splitting a window. Fit the same residual 32-unit tanh MLP on the
first half and evaluate the second; then reverse. Standardize inputs using
the fitting half only (clip standardized values to [-10,10]). Initialize the
residual output to zero so epoch zero exactly recovers the original gate.
Use one fixed 20-epoch schedule, AdamW LR 1e-3, weight decay 1e-3, batch
8,192, seed 176017. Optimize only the training-half target NLL of the
normalized neural/count mixture. Do not select epochs, features, hyperparameters
or the better direction using validation outcomes. The two held-out halves
are combined once by summed NLL/total bytes. These coefficients are fitted
to validation answers and therefore **can never be deployed or scored on
test**. The source must log both directions and the combined result.

Decision gate: advance to the expensive, legal train-only proxy-calibration
route **only if** the combined out-of-half diagnostic reaches <=1.35 BPB,
with both independent directions improving over the unchanged Stage143 gate.
Otherwise stop this gate architecture: its measured signal is too small for
the requested target, even under the optimistic condition of fitting to
validation rather than held-out training text. Keep Stage143 protected either
way. This gate does not establish resource compliance or a new submission
score.

Strongest objection: contiguous halves may differ in topic, and validation
fit itself is not a legal development method. The split is fixed before
measurement; reporting both directions shows instability, and a passing
diagnostic authorizes only a separate train-only fitting experiment.

## Complete result and stop decision (26 September 2026, UTC+8)

The unchanged Stage143 checkpoint and graph reproduced **1.399686162042141
BPB** on all 376,599 validation targets. The target-only neural/count
reconstruction scored **1.399686161937437 BPB**. There were 1,472 independent
windows; the split was fixed after window 736. The diagnostic used 294
causal input features and the fixed 20-epoch/32-hidden-unit schedule.

| Fit half -> held-out half | Held-out targets | NLL change vs Stage143 |
| --- | ---: | ---: |
| First -> second | 188,183 | -22.705185 nats (slight improvement) |
| Second -> first | 188,416 | +308.148304 nats (regression) |

The combined out-of-half score was **1.400044876973115 BPB**, **0.000358715
worse** than Stage143. The source/checkpoint/graph hashes and the summed-NLL
arithmetic are archived at `../results/stage176-evidence/causal-gate-crossfit.json`.
Training-half NLL fell through all 20 epochs in both directions, but the
opposite-half behavior did not transfer. The fixed advancement gate (<=1.35
and both directions improving) **fails decisively**. Do not fit a deployable
train-only proxy version of this MLP, and do not use validation-fitted
coefficients for inference or test. Stage143 remains unchanged.

This result rejects this specific hidden-state/confidence MLP diagnostic;
it does not prove no possible gate can ever exploit the Stage175 oracle gap.
Given the large required 0.049686 gain and the observed transfer failure,
further gate complexity is not the next best use of the finite deadline.
Prioritize a materially stronger **core predictor** or final submission
readiness while retaining all negative evidence.
