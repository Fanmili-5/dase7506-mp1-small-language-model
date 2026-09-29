"""Three alternating fresh-process, validation-only CPU resource comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


CODE = Path(__file__).resolve().parents[1]
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
CANDIDATE_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(path: Path, rows: list[dict]) -> dict:
    if len(rows) != 3:
        raise ValueError("Expected three fresh-process measurements")
    first = rows[0]
    identity = ("checkpoint_sha256", "implementation_sha256",
                "evaluator_sha256", "tokenizer_sha256", "validation_sha256")
    for row in rows:
        if any(row[key] != first[key] for key in identity):
            raise ValueError("Asset changed between repetitions")
        if (row["protocol"] != "7506-mp1-wt2-v2" or row["split"] != "validation"
                or row["precision"] != "fp32" or row["threads"] != 4
                or row["targets"] != 376_599 or row["utf8_bytes"] != 1_148_007
                or row["test_text_opened_by_this_script"] is not False):
            raise ValueError("Invalid validation-only resource protocol")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": first["checkpoint_sha256"],
        "bpb_runs": [row["bpb"] for row in rows],
        "seconds_runs": [row["seconds"] for row in rows],
        "median_seconds": statistics.median(row["seconds"] for row in rows),
        "max_peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    paths = {"baseline": args.baseline.resolve(),
             "candidate": args.candidate.resolve()}
    if sha(paths["baseline"]) != BASELINE_SHA or sha(paths["candidate"]) != CANDIDATE_SHA:
        raise ValueError("Checkpoint identity changed")
    rows = {"baseline": [], "candidate": []}
    started = time.perf_counter()
    for repetition in range(3):
        order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
        for label in order:
            output = subprocess.check_output([
                sys.executable,
                str(CODE / "scripts/cpu_resource_probe_validation_only.py"),
                "--checkpoint", str(paths[label]),
            ], text=True)
            rows[label].append(json.loads(output))
            print(f"Measured {label}, repeat {repetition + 1}/3", flush=True)
    baseline = summarize(paths["baseline"], rows["baseline"])
    candidate = summarize(paths["candidate"], rows["candidate"])
    if any(abs(score - 1.399686162042141) > 2e-5 for score in candidate["bpb_runs"]):
        raise ValueError("Candidate validation BPB changed")
    if any(abs(score - 1.754262747787384) > 2e-5 for score in baseline["bpb_runs"]):
        raise ValueError("Baseline validation BPB changed")
    ratio = candidate["median_seconds"] / baseline["median_seconds"]
    result = {
        "status": "ancillary_post_reboot_validation_only_resource_recheck",
        "protocol": "7506-mp1-wt2-v2",
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "threads": 4,
        "repeats": 3,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_to_baseline_time_ratio": ratio,
        "within_five_x_time_limit": ratio <= 5,
        "within_four_gib_peak_rss_limit": candidate["max_peak_rss_bytes"] <= 4 * 1024**3,
        "memory_scope": "Entire fresh scoring process with validation-only data loading",
        "not_a_replacement_for_official_resource_record": True,
        "test_text_opened_by_these_scripts": False,
        "total_process_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
