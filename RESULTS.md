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
| Final FP32 runtime, unchanged predictor | 1.399686 | [Final Linux validation and resources](code/results/final-evidence/resources-linux-attempt2.json) |

The first pair isolates R-Drop at equal primary-target counts, not equal compute. The backbone comparison also changes width/feed-forward capacity, so it does not isolate convolution. The fixed-count comparison supports neural/count complementarity. Later continuation, calibration and distillation add training cost. The gate gain is validation-selected, not an independent confirmation.

## Complete test

| Predictor | BPB | Targets | Evidence |
| --- | ---: | ---: | --- |
| Initial course baseline | 2.1020149124017866 | 428,405 | [Baseline test](code/results/baseline-test.json) |
| Final predictor | 1.4156576308174311 | 428,405 | [Final full-test JSON](code/results/final-evidence/test.json) |

Both use 1,292,013 raw UTF-8 bytes. The [final freeze record](code/results/final-evidence/freeze.json) binds the inference files before their complete-test reproduction. The original model's [earlier test](code/results/stage143-evidence/test-stage143-20260927.json) was 1.415657616535609 BPB. The runtime update changes execution only, not weights or the prediction method. Its new test record belongs to the exact final package. The final BPB is 32.65% below baseline. The validation-to-test difference cannot distinguish split difficulty from optimism due to repeated validation selection.

## Resources and cost

| Measurement | Result | Evidence |
| --- | --- | --- |
| Final Linux FP32 rerun, four threads, three repeats | 51.018586 s / 10.987902 s baseline; ratio 4.643160 | [Primary resource record](code/results/final-evidence/resources-linux-attempt2.json) |
| Maximum fresh-process RSS on that runner | 2,295,070,720 bytes (2.14 GiB) | Same record |
| Complete inference assets | 55,813,469 bytes (53.23 MiB) | [Final freeze](code/results/final-evidence/freeze.json) |
| Earlier Linux run of identical files, four threads | 28.303853 s / 5.280414 s; ratio 5.360158, above limit | [Earlier resource record](code/results/final-evidence/resources-linux-attempt1.json) |
| Windows, same final execution source, four threads | 74.373097 s / 22.910182 s; ratio 3.246290 | [Windows check](code/results/final-evidence/resources-windows.json) |

I report the final Linux rerun as the primary measurement and retain the earlier failure. Both Linux attempts use identical checkpoint, source, evaluator and tokenizer hashes; each alternates baseline/candidate order across three fresh processes per model. Both expose four logical CPUs and use four intra-op/OpenVINO workers. Default PyTorch inter-op counts differ (two in the earlier run, four in the rerun), but match between the baseline and candidate within each run. CPU model and host contention were not recorded, so the precise cause is unresolved. The Windows checkpoint is separately serialized with identical tensors/configuration; it has a different checkpoint hash.

The timing control has the baseline architecture but a longer-trained checkpoint than the initial accuracy baseline. Training duration does not change that inference graph. The rerun meets all measured limits on its runner, but timing must still be checked on the review machine. The original supplied evaluator defaults to four threads; the guide does not require a separate single-thread Linux pass.

The [main-lineage ledger](code/results/stage143-evidence/lineage-cost.json) records at least 255,225,110 primary-target presentations and 7,754.94 training seconds. The alternate teacher and search add cost. The [archive-wide audit](code/results/project-training-cost-audit.json) records 14.88 training hours across covered runs, including rejected ones. These are lower bounds, not a total hardware bill; setup, evaluation and some failed runs are omitted.

## Provenance

Original JSON records retain their run identifiers, hashes and historical paths. Those paths describe the original experiments and may not exist in this compact checkout. Source snapshots and other exploratory records remain in the [development archive](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/tree/671261f374bd54b28fe6ec3ce6ab890c0b6c72c8). The bundle identifies the clean release commit; it does not present the reorganized branch as the original training checkout.
