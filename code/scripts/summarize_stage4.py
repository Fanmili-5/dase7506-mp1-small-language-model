"""Audit and summarize the Stage-4 scaled-model validation evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_stage3 import ROOT, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage4_summary.json")
    args = parser.parse_args()

    short = audit(ROOT / "runs/stage4-scaled-s17", "scaled", 17, 1200, 300)
    long = audit(ROOT / "runs/stage4-long-scaled-s17", "scaled", 17, 4800, 600)
    modern = audit(ROOT / "runs/stage3-long-modern-s17", "modern", 17, 4800, 600)
    long_directory = ROOT / "runs/stage4-long-scaled-s17"
    long_metrics = json.loads((long_directory / "metrics.json").read_text(encoding="utf-8"))
    best_validation = json.loads(
        (long_directory / "validation_best_cpu_fp32.json").read_text(encoding="utf-8")
    )
    best_checkpoint = long_directory / "checkpoint-best.pt"
    best_digest = hashlib.sha256(best_checkpoint.read_bytes()).hexdigest()
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
    resource_path = ROOT / "results/cpu-stage4-scaled.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8-sig"))

    if resource.get("split") != "validation" or resource.get("device") != "cpu":
        raise ValueError("Unexpected resource benchmark protocol")
    if resource.get("precision") != "fp32" or resource.get("repeats") != 3:
        raise ValueError("Unexpected resource benchmark precision/repeats")
    if resource["candidate"]["checkpoint_sha256"] != short["checkpoint_sha256"]:
        raise ValueError("Resource benchmark did not measure the Stage-4A checkpoint")
    if resource["candidate"]["parameters"] != 3_049_920:
        raise ValueError("Unexpected candidate parameter count")
    if not resource.get("within_five_x_time_limit"):
        raise ValueError("Candidate exceeds the 5x CPU time limit")
    if not resource.get("within_four_gib_peak_rss_limit"):
        raise ValueError("Candidate exceeds the 4 GiB peak-RSS limit")

    asset_paths = [
        best_checkpoint,
        ROOT / "student.py",
        ROOT / "configs/student_scaled.json",
        ROOT / "data/tokenizer.json",
    ]
    asset_bytes = sum(path.stat().st_size for path in asset_paths)
    if asset_bytes > 64 * 1024 ** 2:
        raise ValueError("Core inference assets exceed 64 MiB")

    result = {
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "short_scaled": short,
        "long_scaled": long,
        "validation_selected_scaled": {
            "step": long_metrics["best_validation"]["step"],
            "validation_bpb": best_validation["bpb"],
            "checkpoint_sha256": best_digest,
            "checkpoint_bytes": best_checkpoint.stat().st_size,
        },
        "long_modern_control": modern,
        "short_gain_vs_modern_stage1_bpb": 1.835656527783181 - short["validation_bpb"],
        "long_gain_vs_modern_bpb": modern["validation_bpb"] - long["validation_bpb"],
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
        "caveat": (
            "Validation only. The 1,200-step and 4,800-step checkpoints use different "
            "cosine schedules; compare capacity at matched schedules and target counts only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
