"""Audit and summarize the Stage-6 schedule/dropout screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_stage3 import ROOT, audit


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_best(directory: Path, row: dict) -> dict:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (directory / "validation_best_cpu_fp32.json").read_text(encoding="utf-8")
    )
    checkpoint = directory / "checkpoint-best.pt"
    digest = sha(checkpoint)
    if digest != metrics["best_checkpoint_sha256"]:
        raise ValueError(f"Best-checkpoint training hash mismatch: {directory.name}")
    if digest != validation["checkpoint_sha256"]:
        raise ValueError(f"Best-checkpoint CPU hash mismatch: {directory.name}")
    if validation.get("split") != "validation" or validation.get("device") != "cpu":
        raise ValueError(f"Unexpected best-checkpoint protocol: {directory.name}")
    if validation.get("precision") != "fp32":
        raise ValueError(f"Best checkpoint was not evaluated in FP32: {directory.name}")
    if abs(validation["bpb"] - metrics["best_validation"]["bpb"]) > 1e-6:
        raise ValueError(f"CPU best score disagrees with training record: {directory.name}")
    return {
        "run": row["run"],
        "step": metrics["best_validation"]["step"],
        "validation_bpb": validation["bpb"],
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage6_summary.json")
    args = parser.parse_args()

    specs = [
        ("stage6-schedule3600-w256_d6-s17", "w256_d6", 0.0),
        ("stage6-drop005-w256_d6-s17", "w256_d6_drop005", 0.05),
        ("stage6-drop010-w256_d6-s17", "w256_d6_drop010", 0.10),
    ]
    runs = []
    selected = []
    for name, method, dropout in specs:
        directory = ROOT / "runs" / name
        row = audit(directory, method, 17, 3600, 300)
        if row["parameters"] != 5_247_744:
            raise ValueError(f"Unexpected parameter count: {name}")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics["plan"]["config"].get("dropout") != dropout:
            raise ValueError(f"Unexpected dropout: {name}")
        runs.append(row)
        selected.append(audit_best(directory, row))

    winner_index = min(range(len(selected)), key=lambda index: selected[index]["validation_bpb"])
    winner = selected[winner_index]
    if winner_index != 2:
        raise ValueError("Predeclared Stage-6 winner was not dropout 0.10")

    stage5_path = ROOT.parents[2] / "outputs/windows-stage5-20260919/stage5_summary.json"
    stage5 = json.loads(stage5_path.read_text(encoding="utf-8"))
    stage5_best = stage5["validation_selected_w256_d6"]
    resource = stage5["resource_gate"]

    asset_paths = [
        ROOT / "runs/stage6-drop010-w256_d6-s17/checkpoint-best.pt",
        ROOT / "student.py",
        ROOT / "configs/student_w256_d6_drop010.json",
        ROOT / "data/tokenizer.json",
    ]
    asset_bytes = sum(path.stat().st_size for path in asset_paths)
    if asset_bytes > 64 * 1024**2:
        raise ValueError("Core inference assets exceed 64 MiB")

    result = {
        "split": "validation",
        "device": "cpu",
        "precision": "fp32",
        "matched_3600_step_runs": runs,
        "validation_selected_runs": selected,
        "winner": {**winner, "dropout": 0.10},
        "gain_vs_matched_no_dropout_bpb": selected[0]["validation_bpb"] - winner["validation_bpb"],
        "gain_vs_stage5_best_bpb": stage5_best["validation_bpb"] - winner["validation_bpb"],
        "resource_gate": {
            "basis": "Inherited from Stage 5: identical inference architecture and implementation; dropout is disabled in eval mode.",
            "candidate_to_baseline_time_ratio": resource["candidate_to_baseline_time_ratio"],
            "candidate_max_peak_rss_bytes": resource["candidate_max_peak_rss_bytes"],
            "core_inference_asset_bytes": asset_bytes,
            "within_five_x_time_limit": resource["within_five_x_time_limit"],
            "within_four_gib_peak_rss_limit": resource["within_four_gib_peak_rss_limit"],
            "within_64_mib_core_asset_limit": True,
        },
        "new_training_targets": sum(row["train_targets"] for row in runs),
        "new_train_seconds": sum(row["train_seconds"] for row in runs),
        "new_process_seconds": sum(row["process_seconds"] for row in runs),
        "caveat": (
            "Validation only. This is a matched single-seed dropout screen. The winner "
            "must be replicated before a robust final-model claim."
        ),
    }
    if result["new_training_targets"] != 88_473_600:
        raise ValueError("Stage-6 training budget mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
