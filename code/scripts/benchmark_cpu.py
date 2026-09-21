"""Compare CPU FP32 validation time and fresh-process peak RAM for two checkpoints."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path, runs: list[dict]) -> dict:
    first = runs[0]
    for row in runs:
        for key in ("checkpoint_sha256", "implementation_sha256", "evaluator_sha256", "tokenizer_sha256"):
            if row[key] != first[key]:
                raise ValueError(f"Asset changed between benchmark repetitions: {key}")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": first["checkpoint_sha256"],
        "implementation_sha256": first["implementation_sha256"],
        "checkpoint_bytes": first["checkpoint_bytes"],
        "parameters": first["parameters"],
        "bpb_runs": [row["bpb"] for row in runs],
        "seconds_runs": [row["seconds"] for row in runs],
        "median_seconds": statistics.median(row["seconds"] for row in runs),
        "max_peak_rss_bytes": max(row["peak_rss_bytes"] for row in runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/cpu_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("--repeats and --threads must be positive")

    started = time.perf_counter()
    paths = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    runs = {"baseline": [], "candidate": []}
    for repetition in range(args.repeats):
        # Alternate order to reduce systematic warm-up/thermal ordering effects.
        order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
        for label in order:
            output = subprocess.check_output([
                sys.executable, str(ROOT / "scripts/cpu_resource_probe.py"),
                "--checkpoint", str(paths[label]), "--threads", str(args.threads),
            ], text=True)
            runs[label].append(json.loads(output))
            print(f"Measured {label}, repeat {repetition + 1}/{args.repeats}", flush=True)
    baseline = summarize(paths["baseline"], runs["baseline"])
    candidate = summarize(paths["candidate"], runs["candidate"])
    result = {
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "threads": args.threads,
        "repeats": args.repeats,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_to_baseline_time_ratio": candidate["median_seconds"] / baseline["median_seconds"],
        "within_five_x_time_limit": candidate["median_seconds"] <= 5.0 * baseline["median_seconds"],
        "within_four_gib_peak_rss_limit": candidate["max_peak_rss_bytes"] <= 4 * 1024 ** 3,
        "memory_scope": "Entire fresh scoring process, including imports and fixed data loading",
        "asset_note": "Checkpoint bytes only; the final 64 MiB check must also include all inference assets.",
        "total_process_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
