# Stage214 result: heterogeneous gain is not a lexical hotspot

The fixed train/validation-only analysis reproduced Stage143's full
validation **1.3996861620421412 BPB** and the over-budget equal-mixture
**1.3761826641885053 BPB**, a gain of **0.0235034978536359**. All 12
predeclared cells partition the 376,599 targets and their gains sum to the
same complete-validation total. Supplied train, validation and tokenizer
hashes and both cached target-probability hashes matched. No test text or
new test score was used.

| True-target spelling group | Targets | Stage143 mean NLL | Mixture gain (BPB) | Gain per target (nats) |
| --- | ---: | ---: | ---: | ---: |
| ASCII word start | 174,023 | 4.2043 | 0.011484 | 0.05251 |
| ASCII word continuation | 149,628 | 1.7946 | 0.009101 | 0.04840 |
| Other | 52,948 | 2.1457 | 0.002919 | 0.04387 |

The gain per target is broadly similar across these three groups. The
57,247 medium-frequency targets absent from their causal prefix contribute
**0.004111 BPB**, only **17.49%** of the mixture's gain, versus **15.20%**
of its target count. Stage143's losses are large on those targets, but the
measured larger-backbone complement does not disproportionately repair them.
The exact 33,714 ASCII word-start, medium-frequency, unseen targets have
mean NLL **6.11849**, but their mixture gain is only **0.002611 BPB**.

**Decision:** the predeclared word-internal concentration hypothesis is not
supported. Do not launch another nearby word-prefix, spelling or lexical
output-head pilot on the premise that it can capture most of the known
heterogeneous gain. The remaining difference is distributed across the
sequence and target types, so any new path must change general contextual
modeling or produce a genuinely compact complementary backbone. Existing
Stage197/206/212/213 failures make such a path uncertain; this diagnostic
does not identify a winning replacement or lower the student's acceptance
target. Stage143 remains protected and below-target. The true-target groups
are hindsight diagnostics, never permissible inference gates.

Raw counts and hashes: [`../results/stage214-evidence/lexical-allocation.json`](../results/stage214-evidence/lexical-allocation.json).
