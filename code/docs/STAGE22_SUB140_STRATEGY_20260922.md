# Stage22: strategy toward validation BPB below1.4

The current qualified model is1.473335240 at4.866243x measured CPU. Reaching
1.4 requires0.073335 BPB, about4.98% of the current BPB. This is a stretch goal,
not a promised outcome. Test remains untouched; development uses validation.

## Divergent candidates

1. Extend the fixed neural/count interpolation grid beyond.10.
2. Train a tiny input-only gate between the same neural and count experts.
3. Improve the count expert with train-held-out discount/order calibration.
4. Distil the fixed hybrid into the Transformer during training.
5. Add training-only intermediate-layer next-token supervision.
6. Add training-only multi-future-token heads, removed at inference.
7. Tune stronger embedding/token corruption to reduce the observed generalization gap.
8. Stochastic depth/LayerDrop during training, unchanged inference.
9. Reallocate Transformer depth/width under the measured CPU envelope.
10. Weight-space interpolation between compatible F/H training trajectories.
11. Revisit a recurrent/linear-time backbone with more capacity under the CPU cap.
12. Optimize backbone inference to create room for a ninth Transformer layer.

## Ranked shortlist

| Rank | Candidate | Potential | Cost/risk | Decision |
|---|---|---|---|---|
| 1 | Input-only count gate | Low-medium, fast signal | train leakage/overfit unless calibrated | Measure ceiling first |
| 2 | Intermediate next-token supervision | Medium | one full GPU run; clean ablation | Primary training candidate |
| 3 | Train-only count calibration | Medium | estimator complexity, asset growth | Follow if gate ceiling supports |
| 4 | Training-only future-token heads | Medium-high | fairness/accounting and tuning risk | Secondary training candidate |
| 5 | More capacity after CPU optimization | Medium-high | currently too little timing margin | Park pending larger speed gain |

Fine interpolation and weight soups are cheap but unlikely to close0.073 alone.
A learned gate is only worthwhile if the unchanged experts have enough oracle
complementarity. Stage22 first records a fixed.025-spaced weight grid and an
explicitly illegal target-conditioned oracle. The oracle is never exported or
used as an inference feature. If even that ceiling is above1.4, gating these two
experts cannot meet the requested target, and training effort moves to the
Transformer rather than making a more elaborate gate.

Primary training hypothesis if gating is insufficient: attach lightweight
training-only prediction heads to selected intermediate layers, all predicting
the same legal next-token target. This gives deep supervision without extra
unique target labels or inference cost. Export must remove the auxiliary heads
and reproduce the base prefix-copy architecture exactly. Compare against H with
the same seed, optimizer updates and58,982,400 primary targets; start with one
fixed auxiliary weight and layer set, not a broad validation sweep.

AI assistance is substantive in ideation, implementation, testing and experiment
orchestration. All failed candidates and search cost remain reportable.

Initial fixed grid0--.40 confirmed.10 as best:1.4733350552; .075 gives
1.4733728687 and .125 gives1.4739374457. Fine mixture tuning is closed. An
answer-conditioned per-target oracle over that partial grid reaches only
1.4097420269. Because weights above.40 were absent, this is not yet a strict
two-expert oracle; the diagnostic is rerun through the pure-count endpoint.
Regardless, a deployable input-only gate will be worse than the oracle, so the
main effort moves to improving the neural expert.

The completed0--1 grid retains.10 as the fixed optimum. Its unattainable
per-target oracle reaches1.3614907205: there is theoretically enough expert
complementarity to cross1.4, but a legal gate must infer the choice from input
features and will be worse. Therefore gate work remains a secondary path while
one fixed neural training intervention runs.

## Fixed primary experiment

`student_deep_supervision.py` keeps H's eight-layer prefix-copy Transformer and
embedding-row dropout. After layers4 and6, separate training-only RMSNorms feed
the same tied vocabulary head and predict the same next token. Total auxiliary
loss weight is.20. The final prefix-copy NLL remains the primary loss. Seed17,
batch32,7200 updates, sampling RNG, AdamW settings, LR schedule and58,982,400
unique primary targets match H. Two auxiliary presentations of those same
labels are disclosed separately; no future-label head is used in this experiment.

The auxiliary norms have no RNG-dependent initialization and are removed at
export. The deployed state and outputs must exactly match the original
`student_structured` graph. Fixed last-five checkpoints6000--7200 are averaged,
then independently scored on CPU validation. No automatic test scoring. A gain
would still require rebuilding the train-count hybrid and a fresh resource gate;
failure leaves Stage21 untouched.
