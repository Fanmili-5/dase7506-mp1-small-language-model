"""Audit and summarize the predeclared stage-3 validation experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory: Path, method: str, seed: int, steps: int, eval_every: int) -> dict:
    metrics = json.loads((directory / "metrics.json").read_text())
    validation = json.loads((directory / "validation_cpu_fp32.json").read_text())
    plan = metrics["plan"]
    expected = {
        "implementation": "model" if method == "baseline" else "student",
        "seed": seed, "steps": steps, "micro_batch_size": 32, "grad_accum": 1,
        "learning_rate": 0.001, "min_lr_ratio": 0.1, "warmup_steps": 100,
        "schedule": "baseline", "weight_decay": 0.1, "betas": [0.9, 0.999],
        "grad_clip": 1.0, "precision": "bf16", "device_type": "cuda",
        "threads": 4, "eval_every": eval_every, "eval_batch_size": 32,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected training recipe: {directory.name}")
    targets = steps * 32 * 256
    if metrics["train_tokens"] != targets or metrics["planned_train_targets"] != targets:
        raise ValueError(f"Unexpected target budget: {directory.name}")
    if validation.get("split") != "validation" or validation.get("device") != "cpu" or validation.get("precision") != "fp32":
        raise ValueError(f"Unexpected evaluation protocol: {directory.name}")
    if (validation.get("targets"), validation.get("utf8_bytes")) != (376_599, 1_148_007):
        raise ValueError(f"Incomplete validation split: {directory.name}")
    checkpoint = directory / "checkpoint.pt"
    digest = sha(checkpoint)
    if digest != metrics["checkpoint_sha256"] or digest != validation["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {directory.name}")
    if checkpoint.stat().st_size != metrics["checkpoint_bytes"]:
        raise ValueError(f"Checkpoint size mismatch: {directory.name}")
    config = ROOT / ("configs/baseline.json" if method == "baseline" else f"configs/student_{method}.json")
    implementation = ROOT / ("model.py" if method == "baseline" else "student.py")
    checks = {
        "config_sha256": sha(config), "implementation_sha256": sha(implementation),
        "trainer_sha256": sha(ROOT / "train_experiment.py"),
        "data_manifest_sha256": sha(ROOT / "data/manifest.json"),
    }
    if any(metrics.get(key) != value for key, value in checks.items()):
        raise ValueError(f"Training source mismatch: {directory.name}")
    if validation["implementation_sha256"] != checks["implementation_sha256"]:
        raise ValueError(f"Scored implementation mismatch: {directory.name}")
    if validation["evaluator_sha256"] != sha(ROOT / "evaluate.py") or validation["tokenizer_sha256"] != sha(ROOT / "data/tokenizer.json"):
        raise ValueError(f"Evaluator/tokenizer mismatch: {directory.name}")
    curve = [row for row in metrics["validation_history"] if row["kind"] == "periodic"]
    expected_steps = list(range(eval_every, steps + 1, eval_every))
    if [row["step"] for row in curve] != expected_steps:
        raise ValueError(f"Incomplete validation curve: {directory.name}")
    return {
        "run": directory.name, "method": method, "seed": seed, "steps": steps,
        "train_targets": targets, "parameters": metrics["parameters"],
        "validation_bpb": validation["bpb"], "checkpoint_sha256": digest,
        "checkpoint_bytes": metrics["checkpoint_bytes"], "train_seconds": metrics["train_seconds"],
        "process_seconds": metrics["process_seconds"], "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage3_summary.json")
    args = parser.parse_args()
    short = [
        audit(ROOT / "runs/stage2-rope_swiglu-s17", "rope_swiglu", 17, 1200, 300),
        audit(ROOT / "runs/stage3-rep-rope_swiglu-s23", "rope_swiglu", 23, 1200, 300),
        audit(ROOT / "runs/stage3-rep-rope_swiglu-s42", "rope_swiglu", 42, 1200, 300),
    ]
    long = [
        audit(ROOT / "runs/stage3-long-baseline-s17", "baseline", 17, 4800, 600),
        audit(ROOT / "runs/stage3-long-modern-s17", "modern", 17, 4800, 600),
        audit(ROOT / "runs/stage3-long-rope_swiglu-s17", "rope_swiglu", 17, 4800, 600),
    ]
    scores = [row["validation_bpb"] for row in short]
    result = {
        "split": "validation", "device": "cpu", "precision": "fp32",
        "rope_swiglu_replication": {
            "seeds": [row["seed"] for row in short], "bpb": scores,
            "mean_bpb": statistics.mean(scores), "sample_std_bpb": statistics.stdev(scores),
        },
        "long_runs": long,
        "new_training_targets": sum(row["train_targets"] for row in short[1:] + long),
        "new_train_seconds": sum(row["train_seconds"] for row in short[1:] + long),
        "new_process_seconds": sum(row["process_seconds"] for row in short[1:] + long),
        "caveat": "Validation only. Short and long 1200-step points use different cosine schedules and are not an equal-recipe comparison. Three seeds are descriptive replication, not a rank guarantee.",
    }
    if result["new_training_targets"] != 137_625_600:
        raise ValueError("Stage-3 budget mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
