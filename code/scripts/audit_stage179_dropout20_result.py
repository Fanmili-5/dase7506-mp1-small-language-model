"""Audit collected Stage179 validation evidence without reading test data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL_BPB = 1.4285940451894024
ADVANCEMENT_BPB = CONTROL_BPB - 0.015
TARGETS = 376_599
BYTES = 1_148_007


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(evidence: Path, exported_sha: str) -> dict:
    run = read_json(evidence / "run.json")
    metrics = read_json(evidence / "metrics.json")
    progress = read_json(evidence / "progress.json")
    validation = read_json(evidence / "average-validation-cpu-fp32.json")
    status = read_json(evidence / "status.json")
    if status["status"] != "completed" or metrics["status"] != "completed_training_validation_only":
        raise ValueError("Stage179 job did not finish cleanly")
    if run["source_hashes"] != metrics["source_hashes"]:
        raise ValueError("Run and metrics source manifests differ")
    for name, expected in run["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source mismatch: {name}")
    if (run["protocol"] != "7506-mp1-wt2-v2"
            or metrics["protocol"] != run["protocol"]
            or validation["protocol"] != run["protocol"]
            or not run["no_test_scoring"]
            or run["seed"] != 17
            or run["steps"] != metrics["steps"] != progress["completed_steps"] != 7200
            or run["primary_targets"] != metrics["train_tokens"] != 58_982_400
            or run["precision"] != "bf16"):
        raise ValueError("Training protocol, coverage or test guard mismatch")
    if len(metrics["validation_history"]) != 24 or len(progress["validation_history"]) != 24:
        raise ValueError("Missing one or more scheduled validation points")
    if metrics["validation_history"] != progress["validation_history"]:
        raise ValueError("Progress and final validation histories differ")
    if (validation["split"] != "validation"
            or validation["device"] != "cpu"
            or validation["precision"] != "fp32"
            or validation["targets"] != TARGETS
            or validation["utf8_bytes"] != BYTES
            or validation["checkpoint_sha256"] != exported_sha
            or validation["evaluator_sha256"] != digest(ROOT / "evaluate.py")
            or validation["tokenizer_sha256"] != digest(ROOT / "data/tokenizer.json")
            or validation["implementation_sha256"] != digest(ROOT / "student_hybrid_conv_structured.py")):
        raise ValueError("Independent CPU score identity/coverage mismatch")
    for row in metrics["validation_history"]:
        if (row["targets"] != TARGETS or row["utf8_bytes"] != BYTES
                or row["split"] != "validation" or row["precision"] != "fp32"):
            raise ValueError("Incomplete/interchanged training-time validation")
    score = float(validation["bpb"])
    return {
        "status": "advances_to_next_stage" if score <= ADVANCEMENT_BPB else "stops_at_quality_gate",
        "stage179_cpu_fp32_average_bpb": score,
        "same_target_stage54_control_bpb": CONTROL_BPB,
        "bpb_gain": CONTROL_BPB - score,
        "prespecified_max_bpb_to_advance": ADVANCEMENT_BPB,
        "completed_steps": 7200,
        "primary_training_targets": 58_982_400,
        "validation_targets": TARGETS,
        "validation_utf8_bytes": BYTES,
        "exported_checkpoint_sha256": exported_sha,
        "source_files_verified": len(run["source_hashes"]),
        "test_scored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--exported-checkpoint-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence, args.exported_checkpoint_sha256), indent=2))


if __name__ == "__main__":
    main()
