"""Read-only Stage186 endpoint audit; no test scoring or model promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/stage186-evidence"
CONTROL = ROOT / "results/stage54-evidence/metrics.json"
CONFIG = ROOT / "configs/stage186_narrow_parallel_rdrop.json"
CONTROL_CONFIG = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
TARGETS = 2400 * 32 * 256
MIN_GAIN = 0.020


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite audit")
    config, control_config = read_json(CONFIG), read_json(CONTROL_CONFIG)
    if (config.get("parallel_heads") != 2 or config.get("parallel_head_dim") != 36
            or {key: value for key, value in config.items()
                if key not in {"parallel_heads", "parallel_head_dim"}} != control_config):
        raise ValueError("Unexpected config difference")
    preflight = read_json(EVIDENCE / "preflight.json")
    if (preflight["eligible_for_pilot"] is not True
            or preflight["test_scored"] is not False
            or preflight["config_sha256"] != sha(CONFIG)
            or preflight["parallel_module_sha256"] != sha(
                ROOT / "student_stage186_narrow_parallel.py")):
        raise ValueError("Preflight does not authorize this pilot")
    run, metrics = read_json(EVIDENCE / "run.json"), read_json(EVIDENCE / "metrics.json")
    status = read_json(EVIDENCE / "status.json")
    if (status["status"] != "completed"
            or run["status"] != "training"
            or metrics["status"] != "completed_training_validation_only"
            or run["source_hashes"] != metrics["source_hashes"]
            or metrics["no_test_scoring"] is not True
            or metrics["steps"] != 2400
            or metrics["train_tokens"] != TARGETS
            or metrics["primary_targets"] != TARGETS
            or metrics["seed"] != 17
            or metrics["implementation_sha256"] != sha(
                ROOT / "student_stage186_narrow_parallel_rdrop.py")):
        raise ValueError("Incomplete or changed pilot")
    for relative, expected in metrics["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Source changed after run: {relative}")
    checkpoint_sha = args.remote_checkpoint_sha256.lower()
    if checkpoint_sha != metrics["checkpoint_sha256"]:
        raise ValueError("Remote checkpoint hash differs from metrics")
    validations = metrics["validation_history"]
    if [row["step"] for row in validations] != list(range(300, 2401, 300)):
        raise ValueError("Missing scheduled complete validation")
    if metrics["final_validation"] != validations[-1]:
        raise ValueError("Final validation is not fixed endpoint")
    for row in validations:
        if row["targets"] != 376599 or row["utf8_bytes"] != 1148007:
            raise ValueError("Incomplete validation coverage")
    control = read_json(CONTROL)
    controls = [row for row in control["validation_history"] if row["step"] == 2400]
    if len(controls) != 1:
        raise ValueError("Missing matched Stage54 endpoint")
    control_bpb = controls[0]["bpb"]
    pilot_bpb = validations[-1]["bpb"]
    gain = control_bpb - pilot_bpb
    result = {
        "stage": 186,
        "status": "rejected_below_predeclared_quality_gate" if gain < MIN_GAIN else "passed_pilot_gate_only",
        "matched_control_stage54_step2400_bpb": control_bpb,
        "pilot_step2400_bpb": pilot_bpb,
        "improvement_bpb": gain,
        "required_improvement_bpb": MIN_GAIN,
        "full_validation_targets": validations[-1]["targets"],
        "full_validation_utf8_bytes": validations[-1]["utf8_bytes"],
        "primary_train_targets": metrics["train_tokens"],
        "train_seconds": metrics["train_seconds"],
        "peak_cuda_allocated_gb": metrics["peak_allocated_gb"],
        "peak_cuda_reserved_gb": metrics["peak_reserved_gb"],
        "checkpoint_sha256_remote": checkpoint_sha,
        "metrics_sha256": sha(EVIDENCE / "metrics.json"),
        "preflight_sha256": sha(EVIDENCE / "preflight.json"),
        "no_test_scoring": True,
        "note": "Pilot is not a complete CPU predictor or submission score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
