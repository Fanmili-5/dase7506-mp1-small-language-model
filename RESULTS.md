# Results and evidence

Development comparisons use complete validation. Test is reported only for the frozen final predictor and the original baseline.

| Comparison | Validation BPB | Evidence |
| --- | ---: | --- |
| Ordinary objective, 58,982,400 primary targets | 1.464994 | [Reference](code/results/stage26-evidence/average-validation-cpu-fp32.json) |
| Same network/targets, add R-Drop | 1.450613 | [R-Drop](code/results/stage47-evidence/average-validation-cpu-fp32.json) |
| Attention/convolution backbone | 1.428594 | [Hybrid](code/results/stage54-evidence/average-validation-cpu-fp32.json) |
| Hybrid plus fixed-weight MKN | 1.423206 | [Count scan](code/results/stage55-evidence/scan.json) |
| Later distilled neural model plus fixed-weight order-six MKN | 1.401288 | [Fixed mixture](code/results/stage100-evidence/result.json) |
| Same later experts, selected gate | 1.399686 | [Gate selection](code/results/stage102-evidence/result.json) |
| Frozen CPU implementation | 1.399686 | [Qualification](code/results/stage143-evidence/final.json) |

The first pair isolates R-Drop at equal primary-target counts, not equal compute. The backbone comparison also changes width/feed-forward capacity, so it does not isolate convolution. The fixed-count comparison supports neural/count complementarity. Later continuation, calibration and distillation add training cost. The gate gain is validation-selected, not an independent confirmation.

## Complete test

| Predictor | BPB | Targets | Evidence |
| --- | ---: | ---: | --- |
| Initial course baseline | 2.1020149124017866 | 428,405 | [Baseline test](code/results/baseline-test.json) |
| Frozen final predictor | 1.415657616535609 | 428,405 | [Final test](code/results/stage143-evidence/test-stage143-20260927.json) |

Both use 1,292,013 raw UTF-8 bytes. The [freeze record](code/results/stage143-evidence/freeze-stage143-20260927.json) binds final inference files before test. The final BPB is 32.65% below baseline. The validation-to-test difference cannot distinguish split difficulty from optimism due to repeated validation selection.

## Resources and cost

| Measurement | Result | Evidence |
| --- | --- | --- |
| Windows FP32, four threads, three repeats | 86.0511 s / 23.7861 s baseline; ratio 3.617702 | [Resources](code/results/stage143-evidence/resources.json) |
| Maximum fresh-process RSS | 2,176,729,088 bytes | Same record |
| Complete inference assets | 55,810,412 bytes | [Qualification](code/results/stage143-evidence/final.json) |
| Linux FP32, one thread | ratio 5.502555; exceeds 5× | [Cross-host audit](code/results/stage170-linux-evidence/stage170-linux-resource-audit.json) |

The timing control has the baseline architecture but a longer-trained checkpoint than the initial accuracy baseline. Its training duration does not change the inference graph. A Windows pass does not guarantee a pass on the course CPU.

The [main-lineage ledger](code/results/stage143-evidence/lineage-cost.json) records at least 255,225,110 primary-target presentations and 7,754.94 training seconds. The alternate teacher and search add cost. The [archive-wide audit](code/results/project-training-cost-audit.json) records 14.88 training hours across covered runs, including rejected ones. These are lower bounds, not a total hardware bill; setup, evaluation and some failed runs are omitted.

## Provenance

Original JSON records retain their run identifiers, hashes and historical paths. Those paths describe the original experiments and may not exist in this compact checkout. Source snapshots and other exploratory records remain in the [development archive](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/tree/671261f374bd54b28fe6ec3ce6ab890c0b6c72c8). The bundle identifies the clean release commit; it does not present the reorganized branch as the original training checkout.
