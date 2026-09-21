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
