# Results and evidence

Development comparisons use complete validation. Test is reported only for the frozen final predictor and the original baseline.

| Comparison | Validation BPB | Evidence |
| --- | ---: | --- |
| Auxiliary objectives without R-Drop, 58,982,400 primary targets | 1.464994 | [Comparison records](code/results/comparisons.json) |
| Same network/targets, add R-Drop | 1.450613 | Same records |
| Attention/convolution backbone | 1.428594 | Same records |
| Hybrid plus fixed-weight MKN | 1.423206 | Same records |
| Later distilled neural model plus fixed-weight order-six MKN | 1.401288 | Same records |
| Same later experts, selected gate | 1.399686 | Same records |
| Final FP32 runtime, unchanged predictor | 1.399686 | [Final Linux validation and resources](code/results/final-evidence/resources-linux-attempt2.json) |

The first pair isolates R-Drop at equal primary-target counts, not equal compute. The backbone comparison also changes width/feed-forward capacity, so it does not isolate convolution. The fixed-count comparison supports neural/count complementarity. Later continuation, calibration and distillation add training cost. The gate gain is validation-selected, not an independent confirmation.

## Complete test

| Predictor | BPB | Targets | Evidence |
| --- | ---: | ---: | --- |
| Initial course baseline | 2.1020149124017866 | 428,405 | [Baseline test](code/results/baseline-test.json) |
| Final predictor | 1.4156576308174311 | 428,405 | [Final full-test JSON](code/results/final-evidence/test.json) |

Both use 1,292,013 raw UTF-8 bytes. The [final freeze record](code/results/final-evidence/freeze.json) binds the inference files before their complete-test reproduction. The final BPB is 32.65% below baseline. The validation-to-test difference cannot distinguish split difficulty from optimism due to repeated validation selection.

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

The [training-cost summary](code/results/training-cost.json) records at least 255,225,110 primary-target presentations and 7,754.94 training seconds for the accepted neural lineage. The alternate teacher and search add cost. Across the archived runs, including rejected ones, recorded training time sums to 14.88 hours. These are lower bounds, not a total hardware bill; setup, evaluation and some failed runs are omitted.

## Provenance

The comparison and cost summaries link to the original records at a fixed Git commit and include their SHA-256 hashes. Full search logs and intermediate checkpoints are not part of this submission. The final measurement files remain unchanged, including the slower Linux timing run. The bundle manifest identifies the submitted code and model files.
