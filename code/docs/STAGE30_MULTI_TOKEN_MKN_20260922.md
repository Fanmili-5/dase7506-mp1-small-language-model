# Stage30: multi-token neural plus modified Kneser-Ney

Stage26 improved the unchanged deployed Transformer from 1.4702546457 to
1.4649939083 validation BPB using training-only multi-token prediction. Stage30
kept that checkpoint fixed and rescanned the already-built Stage25 minimum-count
2 modified Kneser-Ney expert at weights 0 through 0.25. No new gradient targets,
count fitting, test scoring or resource claim was introduced.

The fixed scan again selected count weight 0.125. The serialized hybrid scored
**1.4488982961 BPB** under an independent CPU FP32 evaluation, with checkpoint
SHA-256 `3847ff751554be8f22e71882046e91924db590b2bac4df1ac489796625fee817`.
This improves the resource-qualified Stage27 result, 1.4543904321, by
0.0054921360 BPB. Exact sparse-recurrence export and CPU/RAM qualification are
still pending, so Stage27 remains the qualified fallback. No test data was
scored.
