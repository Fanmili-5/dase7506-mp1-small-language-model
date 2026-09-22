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
for the 1.454390432 modified-Kneser-Ney hybrid. `stage39-evidence/` and
`stage39-job-logs/` contain the full-output equivalence, independent CPU FP32
score and fresh three-repeat resource gate for the current 1.446462019
validation leader. Stage27 remains a wider-margin fallback.
