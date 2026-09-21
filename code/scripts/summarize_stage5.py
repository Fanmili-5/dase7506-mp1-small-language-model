"""Audit and summarize the Stage-5 capacity-screen evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_stage3 import ROOT, audit


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage5_summary.json")
    args = parser.parse_args()

    short = [
        audit(ROOT / "runs/stage5-w192_d8-s17", "w192_d8", 17, 1200, 300),
        audit(ROOT / "runs/stage5-w224_d6-s17", "w224_d6", 17, 1200, 300),
        audit(ROOT / "runs/stage5-w256_d6-s17", "w256_d6", 17, 1200, 300),
    ]
    long = audit(ROOT / "runs/stage5-long-w256_d6-s17", "w256_d6", 17, 4800, 600)

    long_directory = ROOT / "runs/stage5-long-w256_d6-s17"
    long_metrics = json.loads((long_directory / "metrics.json").read_text(encoding="utf-8"))
    best_validation = json.loads(
        (long_directory / "validation_best_cpu_fp32.json").read_text(encoding="utf-8")
    )
    best_checkpoint = long_directory / "checkpoint-best.pt"
    best_digest = sha(best_checkpoint)
    if best_digest != long_metrics["best_checkpoint_sha256"]:
        raise ValueError("Best-checkpoint hash mismatch")
    if best_digest != best_validation["checkpoint_sha256"]:
        raise ValueError("CPU best-checkpoint score uses the wrong checkpoint")
    if best_validation.get("split") != "validation" or best_validation.get("device") != "cpu":
        raise ValueError("Unexpected best-checkpoint evaluation protocol")
    if best_validation.get("precision") != "fp32":
        raise ValueError("Best checkpoint was not evaluated in FP32")
    if abs(best_validation["bpb"] - long_metrics["best_validation"]["bpb"]) > 1e-6:
        raise ValueError("CPU best-checkpoint score disagrees with the training record")

    resource = json.loads(
        (ROOT / "results/cpu-stage5-w256_d6.json").read_text(encoding="utf-8-sig")
    )
    if resource.get("split") != "validation" or resource.get("device") != "cpu":
        raise ValueError("Unexpected resource benchmark protocol")
    if resource.get("precision") != "fp32" or resource.get("repeats") != 3:
        raise ValueError("Unexpected resource benchmark precision/repeats")
    short_wide = short[-1]
    if resource["candidate"]["checkpoint_sha256"] != short_wide["checkpoint_sha256"]:
        raise ValueError("Resource benchmark did not measure the Stage-5A wide checkpoint")
    if resource["candidate"]["parameters"] != 5_247_744:
        raise ValueError("Unexpected candidate parameter count")
    if not resource.get("within_five_x_time_limit"):
        raise ValueError("Candidate exceeds the 5x CPU time limit")
    if not resource.get("within_four_gib_peak_rss_limit"):
        raise ValueError("Candidate exceeds the 4 GiB peak-RSS limit")

    asset_paths = [
        best_checkpoint,
        ROOT / "student.py",
        ROOT / "configs/student_w256_d6.json",
        ROOT / "data/tokenizer.json",
    ]
    asset_bytes = sum(path.stat().st_size for path in asset_paths)
    if asset_bytes > 64 * 1024**2:
        raise ValueError("Core inference assets exceed 64 MiB")

    stage4_path = ROOT.parents[2] / "outputs/windows-stage4-20260919/stage4_summary.json"
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    stage4_best = stage4["validation_selected_scaled"]

    result = {
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "short_capacity_screen": short,
        "long_w256_d6": long,
        "validation_selected_w256_d6": {
            "step": long_metrics["best_validation"]["step"],
            "validation_bpb": best_validation["bpb"],
            "checkpoint_sha256": best_digest,
            "checkpoint_bytes": best_checkpoint.stat().st_size,
        },
        "stage4_validation_selected_control": stage4_best,
        "best_gain_vs_stage4_bpb": stage4_best["validation_bpb"] - best_validation["bpb"],
        "endpoint_gain_vs_stage4_endpoint_bpb": (
            stage4["long_scaled"]["validation_bpb"] - long["validation_bpb"]
        ),
        "resource_gate": {
            "candidate_to_baseline_time_ratio": resource["candidate_to_baseline_time_ratio"],
            "candidate_median_seconds": resource["candidate"]["median_seconds"],
            "baseline_median_seconds": resource["baseline"]["median_seconds"],
            "candidate_max_peak_rss_bytes": resource["candidate"]["max_peak_rss_bytes"],
            "core_inference_asset_bytes": asset_bytes,
            "within_five_x_time_limit": True,
            "within_four_gib_peak_rss_limit": True,
            "within_64_mib_core_asset_limit": True,
        },
        "new_training_targets": sum(row["train_targets"] for row in short) + long["train_targets"],
        "new_train_seconds": sum(row["train_seconds"] for row in short) + long["train_seconds"],
        "new_process_seconds": sum(row["process_seconds"] for row in short) + long["process_seconds"],
        "caveat": (
            "Validation only. The short screen compares shapes under one matched recipe. "
            "The long-run gain is a single-seed result and requires replication before a robust claim."
        ),
    }
    if result["new_training_targets"] != 68_812_800:
        raise ValueError("Stage-5 training budget mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
