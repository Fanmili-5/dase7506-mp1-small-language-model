# Stage24: exact resource qualification of the Stage22 hybrid

The Stage22 neural average scored 1.4702546457 CPU FP32 validation BPB. A fixed
grid selected the unchanged train-only absolute-discount count expert at weight
0.075, producing 1.4618048027 in the original slow implementation. Stage24 did
not fit, tune or retrain anything. It exported the exact same tensors and
statistics through the previously tested collapsed sparse backoff recurrence.

Full-validation dual-model comparison passed with maximum absolute log-probability
error below the fixed tolerance and unchanged serialized tensors. The independently
loaded collapsed checkpoint (SHA256
`03957f175359c9ca3a78d19cd21c86d41cfa9ce4c514fe8877025e9f59270111`)
scored 1.4618049084 BPB. The 1.06e-7 difference from the original arithmetic is
numerical, not a learned-quality change.

Fresh three-repeat Windows CPU FP32 measurement against the fixed baseline:

- median-time ratio: 4.9205953393 (passes <=5x);
- maximum peak RSS: 2,024,538,112 bytes (passes <=4 GiB);
- conservative inference assets: 38,153,954 bytes (passes <=64 MiB);
- checkpoint bytes: 37,998,606;
- validation targets/UTF-8 bytes: 376,599 / 1,148,007.

The measured time margin is only about 1.59%, so this host-specific pass is not
a guarantee for every CPU. Fixed-file checks passed before and after the run.
No test split was scored. Raw equivalence, independent validation, resources and
job logs are retained under `results/stage24-evidence/` and
`results/stage24-job-logs/`. The first scheduled-task attempt was rejected by a
runner allow-list before logs or model computation; the corrected `-b` task is
the completed evidence source.
