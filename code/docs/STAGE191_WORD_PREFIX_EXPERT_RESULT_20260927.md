# Stage191 result: stop the fixed word-prefix expert

This was an exploratory, validation-only test of the [pre-outcome plan](STAGE191_WORD_PREFIX_EXPERT_PLAN_20260927.md), not a deployable predictor or test result. The plan and script were committed before reading this result. The train-derived successor distribution is normalized and uses only the current ASCII word prefix inside each independent 256-token validation window. The fixed Stage143 target-probability cache reproduced **1.399686162 BPB** over all **376,599** targets and **1,148,007** raw bytes. No test text was opened by the diagnostic script.

| Fixed weight | Full validation BPB | Gain versus Stage143 | Half 1 / half 2 NLL gain |
| ---: | ---: | ---: | ---: |
| 0 | 1.399686162 | 0 | 0 / 0 nats |
| 0.05, sensitivity only | 1.399508554 | +0.000177608 | +1.28 / +140.05 nats |
| **0.10, prespecified primary** | **1.402632745** | **-0.002946583** | **-1295.25 / -1049.46 nats** |
| 0.20, sensitivity only | 1.413128219 | -0.013442057 | -5565.64 / -5130.71 nats |

The table covered 3,057,014 train-prefix observations, 118,503 distinct prefixes and 583,580 prefix/successor pairs. The fixed minimum-support rule activated at 300,020 validation targets; the actual successor had been seen in train at 264,043 of those. High target incidence therefore did not translate to improved probabilities: sparse relative frequencies were too confident or redundant with Stage143. This is an inference from this fixed experiment, not proof that every morphology-aware model would fail.

The primary cell missed both the **0.015 BPB** full-validation gain and **2,000-nat gain in each half** requirements. The 0.05 sensitivity cell was about 84 times smaller than the full gain gate and essentially flat in the first half. **Stop Stage191:** do not tune a nearby weight or support threshold, build a deployable table, claim a resource pass, promote a checkpoint or inspect test. Stage143 remains the sole Windows-resource-qualified development candidate at 1.399686162 validation BPB, above the aspirational 1.35 target.

The fixed complete record, hashes, coverage and all cells are in [`../results/stage191-evidence/diagnostic.json`](../results/stage191-evidence/diagnostic.json). The same validation set has informed many earlier stages, so even a small positive diagnostic would not have been independent confirmation. This negative result closes only the prespecified ASCII word-prefix count mixture.
