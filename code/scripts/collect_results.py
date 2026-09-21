"""Collect completed run metrics into a deterministic CSV for the report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "implementation",
    "config_path",
    "seed",
    "parameters",
    "train_tokens",
    "precision",
    "train_seconds",
    "validation_bpb",
    "validation_token_ppl",
    "validation_seconds",
    "checkpoint_bytes",
    "checkpoint_sha256",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()
    rows = []
    for metrics_path in sorted(args.runs.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation = metrics.get("validation", {})
        cpu_validation_path = metrics_path.parent / "validation_cpu_fp32.json"
        if cpu_validation_path.exists():
            validation = json.loads(cpu_validation_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": metrics_path.parent.name,
                "implementation": metrics.get("implementation"),
                "config_path": metrics.get("config_path"),
                "seed": metrics.get("seed"),
                "parameters": metrics.get("parameters"),
                "train_tokens": metrics.get("train_tokens"),
                "precision": metrics.get("precision"),
                "train_seconds": metrics.get("train_seconds"),
                "validation_bpb": validation.get("bpb"),
                "validation_token_ppl": validation.get("token_ppl"),
                "validation_seconds": validation.get("seconds"),
                "checkpoint_bytes": metrics.get("checkpoint_bytes"),
                "checkpoint_sha256": metrics.get("checkpoint_sha256"),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
