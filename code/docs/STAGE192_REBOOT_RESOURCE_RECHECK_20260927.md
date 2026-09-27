# Stage192: post-reboot CPU/RAM recheck (validation only)

Status: procedure fixed before measurements on 27 September 2026 UTC+8.
This is an ancillary replication of resource behavior after the Windows
laptop shut down and restarted. It does not select a model, alter Stage143,
freeze the method, score test, or replace the existing course-script three-run
qualification in `results/stage143-evidence/resources.json`.

Use the same Windows user, baseline checkpoint SHA-256
`2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d`,
Stage143 checkpoint SHA-256
`256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3`,
CPU FP32, four threads, all 376,599 validation targets and 1,148,007 raw
bytes. Perform three fresh-process comparisons in order B/C, C/B, B/C.
Each child calls the unchanged `evaluate.score` and `peak_process_memory`,
but loads only the manifest-verified validation text and fixed tokenizer.
The supplied `common.load_data()` is not called because it eagerly opens
train and test even for a validation score. No test text is opened here.

Report each fresh-process score, scoring seconds and lifetime peak RSS;
compare candidate/baseline median scoring seconds and candidate maximum RSS
against 5x and 4 GiB. Recheck the existing 55,810,412-byte inference asset
audit separately; this procedure does not estimate assets. Flag any identity,
coverage or score drift instead of treating a partial run as a pass. The
different validation-only loading scope means the new RSS figures cannot
substitute for the earlier all-split-loader official resource record.
