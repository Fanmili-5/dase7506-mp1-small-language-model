# Results

`summary.csv` is generated from completed `runs/*/metrics.json` files by:

```bash
python scripts/collect_results.py
```

Raw run directories are intentionally excluded from Git because they contain
large checkpoints and resumable optimizer state. Before final submission, retain
the frozen run metadata and publish the matching checkpoint bundle separately.

No score should be entered manually into this directory without a corresponding
metrics file and checkpoint hash.

`stage24-evidence/` contains the previous qualified candidate's full-output
equivalence receipt, independent CPU FP32 validation score, and fresh
three-repeat CPU/RAM resource measurement. `stage24-job-logs/` records the
completed one-off Windows job. Large checkpoints and per-window arrays remain
ignored by Git; their hashes are recorded in the JSON evidence.

`stage27-evidence/` and `stage27-job-logs/` contain the corresponding receipts
for the later 1.454390432 modified-Kneser-Ney hybrid, which is the current
resource-qualified validation leader. Stage24 remains the wider CPU-margin
fallback.
