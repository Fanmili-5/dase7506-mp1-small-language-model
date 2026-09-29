# Results

I used the complete validation split to compare models. The test table reports the selected model and the course baseline.

| Comparison | Validation BPB | Evidence |
| --- | ---: | --- |
| Auxiliary objectives without R-Drop, 58,982,400 primary targets | 1.464994 | [Comparison records](code/results/comparisons.json) |
| Same network/targets, add R-Drop | 1.450613 | Same records |
| Attention/convolution backbone | 1.428594 | Same records |
| Hybrid plus fixed-weight MKN | 1.423206 | Same records |
| Later distilled neural model plus fixed-weight order-six MKN | 1.401288 | Same records |
| Same later experts, selected gate | 1.399686 | Same records |
| OpenVINO FP32 implementation of the selected model | 1.399686 | [Linux validation and resources](code/results/final-evidence/resources-linux-attempt2.json) |

The first pair isolates R-Drop at equal primary-target counts, not equal compute. The backbone comparison also changes width/feed-forward capacity, so it does not isolate convolution. The fixed-count comparison supports neural/count complementarity. Later continuation, calibration and distillation add training cost. The gate gain is validation-selected, not an independent confirmation.

## Complete test

| Predictor | BPB | Targets | Evidence |
| --- | ---: | ---: | --- |
| Initial course baseline | 2.1020149124017866 | 428,405 | [Baseline test](code/results/baseline-test.json) |
| Hybrid model | 1.4156576308174311 | 428,405 | [Test result](code/results/final-evidence/test.json) |

Both use 1,292,013 raw UTF-8 bytes. I fixed the model and inference files before testing; their hashes are in the [model record](code/results/final-evidence/freeze.json). Test BPB is 32.65% below baseline. The validation-to-test difference cannot distinguish split difficulty from optimism due to repeated validation selection.

## Resources and cost

| Measurement | Result | Evidence |
| --- | --- | --- |
| Linux FP32 rerun, four threads, three repeats | 51.018586 s / 10.987902 s baseline; ratio 4.643160 | [Resource measurement](code/results/final-evidence/resources-linux-attempt2.json) |
| Maximum fresh-process RSS on that runner | 2,295,070,720 bytes (2.14 GiB) | Same record |
| Complete inference assets | 55,813,469 bytes (53.23 MiB) | [File manifest](code/results/final-evidence/freeze.json) |
| Earlier Linux run of identical files, four threads | 28.303853 s / 5.280414 s; ratio 5.360158, above limit | [Earlier resource record](code/results/final-evidence/resources-linux-attempt1.json) |
| Windows, same implementation, four threads | 74.373097 s / 22.910182 s; ratio 3.246290 | [Windows measurement](code/results/final-evidence/resources-windows.json) |

The report uses the Linux rerun as its primary measurement. The earlier run exceeded the time limit despite identical checkpoint, source, evaluator and tokenizer hashes. Each run alternates baseline/model order across three fresh processes per model. Both expose four logical CPUs and use four intra-op/OpenVINO workers. Default PyTorch inter-op counts differ (two in the earlier run, four in the rerun), but match between models within each run. I did not record the CPU model or host contention, so I cannot explain the timing difference precisely. The Windows checkpoint has identical tensors/configuration but was serialized separately and has a different file hash.

The timing control has the baseline architecture but a longer-trained checkpoint than the initial accuracy baseline. Training duration does not change that inference graph. The rerun meets the limits on its runner; timing should be measured again on a different machine. I used the supplied evaluator's default of four threads.

The [training-cost summary](code/results/training-cost.json) records at least 255,225,110 primary-target presentations and 7,754.94 training seconds for the main neural branch. The alternate teacher and search add cost. Recorded training time across the experiments, including rejected ones, sums to 14.88 hours. These are lower bounds because setup, evaluation and some failed runs are omitted.

## Measurement records

The comparison and cost summaries include SHA-256 hashes and links to the original measurements at a fixed Git commit.
