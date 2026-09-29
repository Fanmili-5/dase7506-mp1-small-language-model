# Stage192 result: Windows post-reboot resource check

This is the ancillary result of the [fixed validation-only procedure](STAGE192_REBOOT_RESOURCE_RECHECK_20260927.md), not a new quality candidate, method freeze or test score. The Windows laptop reported AC power online before the run. Three alternating fresh-process CPU FP32 four-thread comparisons completed in the declared B/C, C/B, B/C order. Every baseline run reproduced **1.754262747787384** validation BPB and every unchanged Stage143 run reproduced **1.399686162042141** BPB over all **376,599** targets.

| Metric | Baseline | Stage143 |
| --- | ---: | ---: |
| Scoring seconds, three runs | 20.9851 / 20.4182 / 20.6071 | 87.0359 / 85.8928 / 85.8764 |
| Median scoring seconds | 20.6071 | 85.8928 |
| Maximum whole-process peak RSS | 695,316,480 bytes | 1,599,090,688 bytes |

The candidate/baseline median CPU ratio is **4.168122405**, below 5; candidate peak RSS is below 4 GiB. The 18 qualified inference files were separately hash-audited after restart at **55,810,412 bytes**, below 64 MiB; this Stage192 timing script did not count assets. These observations support that the protected candidate remains usable on this Windows laptop after the shutdown. They do **not** establish compliance on every course host: the independent Linux one-thread run was previously above 5x.

The course-script qualification remains the formal local resource record: it loaded all supplied splits and measured **3.617701668x** time and **2,176,729,088-byte** candidate peak RSS. Stage192 intentionally loaded only validation so it did not open test text; therefore its lower whole-process RSS is not directly comparable and **must not replace** that official-scope record. Different baseline and candidate timing changes across sessions cannot be assigned to a single cause. No parameters, inference assets, method freeze or test score changed.

All six individual process records, identities, coverage, seconds and RSS values are in [`../results/stage192-evidence/reboot-resources.json`](../results/stage192-evidence/reboot-resources.json).
