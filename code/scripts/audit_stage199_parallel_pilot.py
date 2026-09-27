"""Audit fixed Stage199 2,400-step endpoint against the matched Stage54 control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/stage199-evidence"
CONTROL = ROOT / "results/stage54-evidence/metrics.json"
CONFIG = ROOT / "configs/stage177_parallel_mixer_rdrop.json"
TARGETS = 2400 * 32 * 256
MIN_GAIN = 0.030


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
        parser.error("Refusing to overwrite existing audit")
    preflight = read_json(EVIDENCE / "preflight.json")
    if (preflight.get("training_gate_passed") is not True
            or preflight.get("no_data_split_opened") is not True
            or preflight.get("config_sha256") != sha(CONFIG)
            or preflight.get("source_sha256") != sha(
                ROOT / "scripts/preflight_stage199_parallel_gpu.py")):
        raise ValueError("Stage199 fixed-batch preflight is missing or changed")
    run, metrics = read_json(EVIDENCE / "run.json"), read_json(EVIDENCE / "metrics.json")
    if (run.get("status") != "training"
            or metrics.get("status") != "completed_training_validation_only"
            or run.get("source_hashes") != metrics.get("source_hashes")
            or metrics.get("no_test_scoring") is not True
            or metrics.get("steps") != 2400
            or metrics.get("train_tokens") != TARGETS
            or metrics.get("primary_targets") != TARGETS
            or metrics.get("seed") != 17
            or metrics.get("implementation_sha256") != sha(
                ROOT / "student_stage177_parallel_mixer_rdrop.py")):
        raise ValueError("Incomplete or changed Stage199 pilot")
    if "data/wikitext_test.txt" in metrics["source_hashes"]:
        raise ValueError("Test text must not be part of the development run")
    for relative, expected in metrics["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Training source changed after run: {relative}")
    remote_sha = args.remote_checkpoint_sha256.lower()
    if remote_sha != metrics.get("checkpoint_sha256"):
        raise ValueError("Remote checkpoint hash differs from completed metrics")
    validations = metrics["validation_history"]
    if [row["step"] for row in validations] != list(range(300, 2401, 300)):
        raise ValueError("Missing scheduled complete-validation points")
    if metrics["final_validation"] != validations[-1]:
        raise ValueError("Endpoint differs from fixed selection checkpoint")
    for row in validations:
        if row["targets"] != 376599 or row["utf8_bytes"] != 1148007:
            raise ValueError("Incomplete validation coverage")
    control = read_json(CONTROL)
    matched = [row for row in control["validation_history"] if row["step"] == 2400]
    if len(matched) != 1:
        raise ValueError("Missing same-seed, same-target Stage54 endpoint")
    control_bpb = matched[0]["bpb"]
    pilot_bpb = validations[-1]["bpb"]
    gain = control_bpb - pilot_bpb
    result = {
        "stage": 199,
        "status": "passed_pilot_quality_gate_only" if gain >= MIN_GAIN else
                  "rejected_below_predeclared_quality_gate",
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
        "checkpoint_sha256_remote": remote_sha,
        "metrics_sha256": sha(EVIDENCE / "metrics.json"),
        "preflight_sha256": sha(EVIDENCE / "preflight.json"),
        "no_test_scoring": True,
        "note": "GPU pilot is not a complete CPU predictor or submission score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
