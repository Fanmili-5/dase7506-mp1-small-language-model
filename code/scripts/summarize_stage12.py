"""Audit the five preregistered Stage-12 CPU validation candidates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, sha

NAMES = ("checkpoint-best", "checkpoint", "average-last2", "average-last3", "average-last5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--resource", type=Path, help="Optional resource gate for the selected checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    plan = metrics["plan"]
    if (plan["steps"], plan["seed"], metrics["train_tokens"]) != (7200, 17, 58982400):
        raise ValueError("Run does not match the preregistered Stage-12 budget and seed.")
    control = json.loads(args.control.read_text(encoding="utf-8-sig"))

    def check_score(score):
        if (score["protocol"], score["split"], score["precision"], score["device"]) != (
            PROTOCOL, "validation", "fp32", "cpu"
        ):
            raise ValueError("Expected official CPU FP32 validation evidence.")
        if (score["targets"], score["utf8_bytes"]) != (376599, 1148007):
            raise ValueError("Validation coverage differs from the fixed protocol.")
        if not math.isfinite(score["bpb"]):
            raise ValueError("Non-finite validation BPB.")

    check_score(control)
    rows = []
    for name in NAMES:
        path = args.run_dir / f"{name}.pt"
        score = json.loads((args.run_dir / f"{name}-validation-cpu-fp32.json").read_text(encoding="utf-8-sig"))
        check_score(score)
        if sha(path) != score["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint does not match its score: {path}")
        for key in ("evaluator_sha256", "tokenizer_sha256", "implementation_sha256"):
            if score[key] != control[key]:
                raise ValueError(f"Control and candidate differ in {key}.")
        rows.append({
            "checkpoint": path.name, "checkpoint_sha256": score["checkpoint_sha256"],
            "checkpoint_bytes": path.stat().st_size, "validation_bpb": score["bpb"],
            "improvement_vs_control": control["bpb"] - score["bpb"],
        })
    winner = min(rows, key=lambda row: row["validation_bpb"])
    inference_files = [
        args.run_dir / winner["checkpoint"], ROOT / "student.py",
        ROOT / "data/tokenizer.json", ROOT / "common.py", ROOT / "evaluate.py",
        ROOT / "requirements.txt",
    ]
    inference_assets = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in inference_files
    ]
    resource_gate = None
    if args.resource:
        resource = json.loads(args.resource.read_text(encoding="utf-8-sig"))
        if resource["candidate"]["checkpoint_sha256"] != winner["checkpoint_sha256"]:
            raise ValueError("Resource evidence is for a different checkpoint.")
        if (resource["split"], resource["device"], resource["precision"], resource["repeats"]) != (
            "validation", "cpu", "fp32", 3
        ):
            raise ValueError("Resource gate must use three CPU FP32 validation repeats.")
        resource_gate = {
            "evidence_sha256": sha(args.resource),
            "time_ratio": resource["candidate_to_baseline_time_ratio"],
            "peak_rss_bytes": resource["candidate"]["max_peak_rss_bytes"],
            "time_pass": resource["within_five_x_time_limit"],
            "ram_pass": resource["within_four_gib_peak_rss_limit"],
        }
    result = {
        "protocol": PROTOCOL, "split": "validation", "precision": "fp32", "device": "cpu",
        "control_bpb": control["bpb"], "control_checkpoint_sha256": control["checkpoint_sha256"],
        "run": args.run_dir.name, "seed": 17, "planned_steps": 7200,
        "train_targets": metrics["train_tokens"], "train_seconds": metrics["train_seconds"],
        "trainer_sha256": metrics["trainer_sha256"],
        "metrics_sha256": sha(args.run_dir / "metrics.json"),
        "candidates": rows, "selected": winner,
        "advance_threshold_bpb": 0.003,
        "advance_to_replication": winner["improvement_vs_control"] >= 0.003,
        "inference_assets": inference_assets,
        "inference_asset_bytes": sum(asset["bytes"] for asset in inference_assets),
        "within_64_mib_asset_limit": sum(asset["bytes"] for asset in inference_assets) <= 64 * 1024**2,
        "resource_gate": resource_gate,
        "status": "validation_selection_only; replication pending; no new test result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
