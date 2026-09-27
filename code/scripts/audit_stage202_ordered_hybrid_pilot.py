"""Audit the fixed Stage202 development pilot against Stage54 and its gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/stage202-evidence"
CONTROL = ROOT / "results/stage54-evidence/metrics.json"
CONFIG = ROOT / "configs/stage202_bottom_local_top_global.json"
REQUIRED_GAIN = 0.03
TARGETS = 2400 * 32 * 256
REMOTE_CHECKPOINT_SHA = "cf388d9341fdd9c13b349007fd95a6961ffe3ffe5a6ed3ea756a24a49103a135"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> dict:
    preflight = read_json(EVIDENCE / "preflight.json")
    run = read_json(EVIDENCE / "run.json")
    metrics = read_json(EVIDENCE / "metrics.json")
    progress = read_json(EVIDENCE / "progress.json")
    status = read_json(EVIDENCE / "status.json")
    if (preflight.get("training_gate_passed") is not True
            or preflight.get("no_data_split_opened") is not True
            or preflight.get("equal_parameter_count") is not True
            or preflight.get("config_sha256") != sha(CONFIG)
            or preflight.get("source_sha256") != sha(
                ROOT / "scripts/preflight_stage202_ordered_hybrid.py")):
        raise ValueError("Missing or changed Stage202 GPU preflight")
    if (run.get("status") != "training"
            or metrics.get("status") != "completed_training_validation_only"
            or status.get("status") != "completed"
            or metrics.get("no_test_scoring") is not True
            or run.get("source_hashes") != metrics.get("source_hashes")
            or metrics.get("steps") != 2400
            or metrics.get("train_tokens") != TARGETS
            or metrics.get("primary_targets") != TARGETS
            or metrics.get("seed") != 17
            or metrics.get("checkpoint_sha256") != REMOTE_CHECKPOINT_SHA
            or progress.get("completed_steps") != 2400):
        raise ValueError("Incomplete or changed Stage202 development run")
    if "data/wikitext_test.txt" in metrics["source_hashes"]:
        raise ValueError("Test text must not be a training source")
    for relative, expected in metrics["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Changed source after training: {relative}")
    validations = metrics["validation_history"]
    if [row["step"] for row in validations] != list(range(300, 2401, 300)):
        raise ValueError("Missing fixed complete-validation points")
    if metrics["final_validation"] != validations[-1]:
        raise ValueError("Fixed endpoint and final validation differ")
    if any(row["targets"] != 376599 or row["utf8_bytes"] != 1148007
           for row in validations):
        raise ValueError("Incomplete validation coverage")
    control = read_json(CONTROL)
    control_by_step = {row["step"]: row["bpb"]
                       for row in control["validation_history"]}
    pilot_bpb = validations[-1]["bpb"]
    control_bpb = control_by_step[2400]
    gain = control_bpb - pilot_bpb
    return {
        "stage": 202,
        "status": ("passed_fixed_quality_gate_only" if gain >= REQUIRED_GAIN
                   else "rejected_below_predeclared_quality_gate"),
        "matched_control_step2400_bpb": control_bpb,
        "pilot_step2400_bpb": pilot_bpb,
        "gain_bpb": gain,
        "required_gain_bpb": REQUIRED_GAIN,
        "validation_targets": validations[-1]["targets"],
        "validation_utf8_bytes": validations[-1]["utf8_bytes"],
        "primary_train_targets": TARGETS,
        "pilot_minus_control_bpb_by_step": [
            {"step": row["step"], "difference": row["bpb"] - control_by_step[row["step"]]}
            for row in validations
        ],
        "train_seconds": metrics["train_seconds"],
        "peak_cuda_allocated_gb": metrics["peak_allocated_gb"],
        "peak_cuda_reserved_gb": metrics["peak_reserved_gb"],
        "checkpoint_sha256_remote": REMOTE_CHECKPOINT_SHA,
        "metrics_sha256": sha(EVIDENCE / "metrics.json"),
        "scheduled_task_status": status["status"],
        "test_scored": False,
        "fixed_file_checker_hashed_test_file_without_tokenizing_or_scoring": True,
        "note": "A GPU development pilot is not an inference resource qualification or submission score.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing audit")
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
