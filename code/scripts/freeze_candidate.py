"""Create an immutable pre-test manifest for the Stage-9 selected candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage9-summary", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage9 = json.loads(args.stage9_summary.read_text(encoding="utf-8"))
    resource = json.loads(args.resource.read_text(encoding="utf-8-sig"))
    selected = stage9["prospective_final_checkpoint"]
    checkpoint = ROOT / "runs" / selected["run"] / "checkpoint-best.pt"
    config = ROOT / "configs/student_w256_d6_drop010.json"
    assets = {
        "checkpoint": checkpoint,
        "implementation": ROOT / "student.py",
        "config": config,
        "tokenizer": ROOT / "data/tokenizer.json",
        "evaluator": ROOT / "evaluate.py",
        "common": ROOT / "common.py",
    }
    asset_rows = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}
        for name, path in assets.items()
    }
    if asset_rows["checkpoint"]["sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint does not match Stage-9 summary")
    if resource.get("split") != "validation" or resource.get("device") != "cpu":
        raise ValueError("Resource benchmark did not use CPU validation")
    if resource.get("precision") != "fp32" or resource.get("repeats") != 3:
        raise ValueError("Resource benchmark did not use three FP32 repeats")
    if resource["candidate"]["checkpoint_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Resource benchmark used a different candidate")
    if not resource.get("within_five_x_time_limit"):
        raise ValueError("Candidate exceeds the 5x CPU time limit")
    if not resource.get("within_four_gib_peak_rss_limit"):
        raise ValueError("Candidate exceeds the 4 GiB peak-RSS limit")
    core_asset_bytes = sum(asset_rows[name]["bytes"] for name in ("checkpoint", "implementation", "config", "tokenizer"))
    if core_asset_bytes > 64 * 1024**2:
        raise ValueError("Core inference assets exceed 64 MiB")

    endpoint = next(row for row in stage9["endpoint_runs"] if row["seed"] == selected["seed"])
    manifest = {
        "status": "frozen_pre_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "test_evaluated_during_development": False,
        "selection_disclosure": (
            "The lowest CPU FP32 validation checkpoint among predeclared seeds "
            "17, 23, and 42 was selected."
        ),
        "candidate": {
            **selected,
            "method": stage9["recipe"]["method"],
            "parameters": endpoint["parameters"],
            "training_steps": endpoint["steps"],
            "training_targets": endpoint["train_targets"],
            "train_seconds": endpoint["train_seconds"],
            "process_seconds": endpoint["process_seconds"],
        },
        "three_seed_validation": stage9["replication"],
        "resource_gate": {
            "repeats": resource["repeats"],
            "threads": resource["threads"],
            "baseline_median_seconds": resource["baseline"]["median_seconds"],
            "candidate_median_seconds": resource["candidate"]["median_seconds"],
            "candidate_to_baseline_time_ratio": resource["candidate_to_baseline_time_ratio"],
            "candidate_max_peak_rss_bytes": resource["candidate"]["max_peak_rss_bytes"],
            "core_inference_asset_bytes": core_asset_bytes,
            "within_five_x_time_limit": True,
            "within_four_gib_peak_rss_limit": True,
            "within_64_mib_core_asset_limit": True,
        },
        "assets": asset_rows,
        "source_archive": {
            "path": str(args.source_archive),
            "sha256": sha(args.source_archive),
            "bytes": args.source_archive.stat().st_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
