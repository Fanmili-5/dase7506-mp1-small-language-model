# Stage27: modified Kneser-Ney hybrid qualification

Stage25 selected the train-only order-5 modified Kneser-Ney expert with minimum
effective count 2 and a 0.125 mixture weight. Its original implementation scored
1.4543903596 validation BPB. Stage27 changed no tensor, count statistic, pruning
decision or mixture weight; it only reused the collapsed sparse recurrence.

Full-validation equivalence passed with maximum absolute log-probability error
5.7220459e-6. The independently loaded collapsed checkpoint (SHA256
`397b68d7b39c5f751d073d7f5985c2f88ce69de847f30b488e7f056b483ff343`)
scored 1.4543904321 CPU FP32 validation BPB. The 7.25e-8 difference is numerical
arithmetic, not a quality change.

Fresh three-repeat Windows resource measurement:

- median candidate/baseline CPU ratio: 4.9700760245 (passes <=5x);
- candidate/baseline medians: 61.0409678 / 12.2816970 seconds;
- maximum peak RSS: 2,044,981,248 bytes (passes <=4 GiB);
- conservative inference assets: 44,210,786 bytes (passes <=64 MiB);
- validation targets/UTF-8 bytes: 376,599 / 1,148,007.

The timing margin is only about 0.60%, so Stage24 remains a useful safer fallback
and this host-specific pass is not a portability guarantee. Fixed files passed
before and after the task. No test split was scored. Raw evidence and job logs
are under `results/stage27-evidence/` and `results/stage27-job-logs/`.
