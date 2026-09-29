# Stage112 direction screen: faster Transformer under the same scorer

This is a provisional ideation ledger, not a validation result or a new
submission claim. The problem-first constraint is concrete: Stage105 has
1.399686 BPB but takes 5.512× baseline CPU time, roughly 12 seconds beyond
its own 5× gate. The profiler assigns about 63% of a full-batch forward to
matrix multiplications (especially eight SwiGLU FFNs), 9% to attention,
2% to the confidence top-2, and less than 1% to count-key search. Stage85
already spends 4.900× on the same-size Transformer with a simpler count head.

## Divergence: candidate mechanisms

1. Remove one whole Transformer block, then distill/repair its effect on train
   text. This could save enough FFN and attention time; untrained quality loss
   is the first falsification test.
2. Remove only one FFN sub-block, then train-only distill/repair. It saves less
   but may preserve more quality; screen all eight locations before choosing.
3. Learn a cheap train-only surrogate for the Stage102 dynamic gate from hidden
   states. It could replace top-2 and count-confidence work, but the full
   Transformer would still sit near the 5× limit.
4. Use per-token conditional FFN execution. It offers more savings, but sparse
   gather/scatter overhead and error propagation require a separate pilot.
5. Fold Stage94 temperature/prior into existing affine parameters and
   precompute count maximum mass. Exact, but the measured overhead bounds
   suggest it cannot alone save the required ~12 seconds.
6. Replace PyTorch neural features with ONNX Runtime. A 12-repeat FP32 probe
   was slower (2.209 versus 2.123 seconds per batch); reject.
7. TorchScript-optimize neural features. A 12-repeat exact probe was slower
   (2.178 versus 2.151 seconds per batch); reject.
8. SVD-factor the learned FFNs. Rank-192 input matrices had ~35% relative
   Frobenius error; without major retraining, reject.
9. Remove 10–25% low-importance SwiGLU channels. Train-prefix perturbation
   changed feature RMS by 20–36%; reject direct pruning.
10. Further optimize n-gram binary search. The measured search consumes only
    ~0.1% of forward time; reject as primary strategy.
11. Continue teacher-only distillation of Stage92. The completed Stage109
    average regressed to 1.402744; reject this loss balance.
12. Retune only static calibration/MKN weight. Earlier Stage94/99 scans left
    >0.001 BPB to the goal; useful control, unlikely standalone solution.

## Convergence and pilot

The three live directions are (1) whole-block removal with repair, (2)
FFN-only removal with repair, and (3) a low-cost gate surrogate. The simplest
falsification comes first: use the exact Stage92 neural checkpoint, replace
one of its eight blocks or FFNs at a time with an identity/zero residual, and
score full validation with the fixed train-only MKN and unchanged calibration.
No weights or hyperparameters are fitted on validation; it selects only which
structural intervention merits train-only repair. Then:

1. Measure the no-repair BPB change for every location and identify whether
   any removal loses less than a repairable margin.
2. On the best predeclared location, train with supplied train prefixes and a
   fixed teacher/hard-label objective; audit the complete validation result.
3. If quality is competitive, export the exact smaller graph and test the
   same three CPU/RAM/asset resource gates. Test remains untouched.

Two-sentence pitch: the current Transformer meets the desired validation
quality only with a dynamic count gate, but its eight-block computation misses
the CPU limit. We will test whether one low-contribution block can be removed
and repaired using training-text distillation, exchanging redundant depth for
the speed headroom the gate needs.

Strongest objection: a no-repair ablation may look small yet retraining could
overfit or fail to regain the last 0.001 BPB. The pilot reports all locations,
uses a fixed repair budget, and requires a separately serialized, fully
resource-audited predictor before promotion.

## Stage112 measured screen

The fixed Stage92 + Stage73 order-six reference reproduced at **1.401288462
BPB** on all 376,599 validation targets. Removing original block 5 was the
least damaging whole-block intervention: **1.442601479 BPB** (+0.041313017).
Removing only the eighth FFN was best among FFN-only interventions:
**1.424480142 BPB** (+0.023191680). All 16 locations and the unchanged
reference are in `code/results/stage112-evidence/result.json`. These numbers
are untrained interventions, not final candidate scores.

An eight-repeat CPU FP32 microbenchmark of the exact Stage105 gate predictor
found 2.694 seconds median per 32-window batch unmodified, 2.536 seconds with
the eighth FFN omitted (0.941 relative time), and 2.384 seconds with original
block 5 omitted (0.885 relative time). The FFN-only reduction is unlikely to
cover Stage105's 5.512× CPU ratio; whole-block removal has a plausible but
unverified path below 5×. This extrapolation is not a resource qualification.
Stage113 therefore repairs the seven-block architecture with train-prefix
teacher/hard-label learning, then must pass full validation and CPU/RAM/asset
checks before promotion. Test was not scored.
