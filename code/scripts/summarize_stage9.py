"""Audit and summarize frozen Stage-9 three-seed replication, validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from summarize_stage3 import ROOT, audit
from summarize_stage6 import audit_best


SPECS = (
    (17, "stage8-long-drop010-w256_d6-s17"),
    (23, "stage9-long-drop010-w256_d6-s23"),
    (42, "stage9-long-drop010-w256_d6-s42"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage9_summary.json")
    args = parser.parse_args()

    endpoints = []
    selected = []
    for seed, name in SPECS:
        directory = ROOT / "runs" / name
        endpoint = audit(directory, "w256_d6_drop010", seed, 4800, 300)
        if endpoint["parameters"] != 5_247_744:
            raise ValueError(f"Unexpected parameter count: {name}")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics["plan"]["config"].get("dropout") != 0.10:
            raise ValueError(f"Unexpected dropout: {name}")
        endpoints.append(endpoint)
        selected.append({**audit_best(directory, endpoint), "seed": seed})

    scores = [row["validation_bpb"] for row in selected]
    winner = min(selected, key=lambda row: row["validation_bpb"])
    asset_paths = [
        ROOT / "runs" / winner["run"] / "checkpoint-best.pt",
        ROOT / "student.py",
        ROOT / "configs/student_w256_d6_drop010.json",
        ROOT / "data/tokenizer.json",
    ]
    asset_bytes = sum(path.stat().st_size for path in asset_paths)
    if asset_bytes > 64 * 1024**2:
        raise ValueError("Core inference assets exceed 64 MiB")

    stage6_path = ROOT.parents[2] / "outputs/windows-stage6-20260919/stage6_summary.json"
    stage6 = json.loads(stage6_path.read_text(encoding="utf-8"))
    inherited = stage6["resource_gate"]
    result = {
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "recipe": {
            "method": "w256_d6_drop010",
            "steps": 4800,
            "targets_per_seed": 39_321_600,
            "selection_rule": "Lowest CPU FP32 validation checkpoint among seeds 17, 23, and 42.",
        },
        "endpoint_runs": endpoints,
        "validation_selected_runs": selected,
        "replication": {
            "seeds": [row["seed"] for row in selected],
            "bpb": scores,
            "mean_bpb": statistics.mean(scores),
            "sample_std_bpb": statistics.stdev(scores),
        },
        "prospective_final_checkpoint": winner,
        "resource_gate_pending_repeat": {
            "basis": "Inherited architecture-level evidence only; repeat on the selected checkpoint before freeze.",
            "previous_candidate_to_baseline_time_ratio": inherited["candidate_to_baseline_time_ratio"],
            "previous_candidate_max_peak_rss_bytes": inherited["candidate_max_peak_rss_bytes"],
            "current_core_inference_asset_bytes": asset_bytes,
            "within_64_mib_core_asset_limit": True,
        },
        "new_training_targets": sum(row["train_targets"] for row in endpoints if row["seed"] != 17),
        "new_train_seconds": sum(row["train_seconds"] for row in endpoints if row["seed"] != 17),
        "new_process_seconds": sum(row["process_seconds"] for row in endpoints if row["seed"] != 17),
        "caveat": (
            "Validation only. The reported winner is selected across three seeds on validation, "
            "which is explicitly disclosed. Test remains untouched."
        ),
    }
    if result["new_training_targets"] != 78_643_200:
        raise ValueError("Stage-9 new-training budget mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
