"""Audit stage-2 artifacts and summarize all predeclared seeds, validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 23, 42)
METHODS = ("baseline", "rope", "modern")
LADDER = ("rope", "rope_swiglu", "rope_swiglu_rms", "rope_swiglu_rms_nobias", "modern")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(name: str, seed: int) -> dict:
    stage = "stage1" if seed == 17 and name in METHODS else "stage2"
    directory = ROOT / "runs" / f"{stage}-{name}-s{seed}"
    metrics = json.loads((directory / "metrics.json").read_text())
    validation = json.loads((directory / "validation_cpu_fp32.json").read_text())
    plan = metrics["plan"]
    required = {"steps": 1200, "micro_batch_size": 32, "grad_accum": 1,
                "learning_rate": 0.001, "weight_decay": 0.1,
                "schedule": "baseline", "seed": seed, "precision": "bf16",
                "warmup_steps": 100, "min_lr_ratio": 0.1, "grad_clip": 1.0,
                "betas": [0.9, 0.999], "eval_every": 300, "eval_batch_size": 32,
                "threads": 4, "device_type": "cuda",
                "implementation": "model" if name == "baseline" else "student"}
    if any(plan.get(key) != value for key, value in required.items()):
        raise ValueError(f"Unexpected training recipe: {directory.name}")
    if metrics["train_tokens"] != 9_830_400 or metrics["seed"] != seed:
        raise ValueError(f"Unexpected training budget/seed: {directory.name}")
    if metrics["protocol"] != "7506-mp1-wt2-v2" or validation["protocol"] != metrics["protocol"]:
        raise ValueError(f"Protocol mismatch: {directory.name}")
    if (validation["split"], validation["precision"], validation["device"],
        validation["targets"], validation["utf8_bytes"]) != (
            "validation", "fp32", "cpu", 376_599, 1_148_007):
        raise ValueError(f"Unexpected evaluation protocol: {directory.name}")
    if not math.isfinite(validation["bpb"]) or validation["bpb"] <= 0:
        raise ValueError("Invalid validation BPB")
    checkpoint_hash = sha(directory / "checkpoint.pt")
    if checkpoint_hash != metrics["checkpoint_sha256"] or checkpoint_hash != validation["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {directory.name}")
    if metrics["checkpoint_bytes"] != (directory / "checkpoint.pt").stat().st_size:
        raise ValueError(f"Checkpoint size mismatch: {directory.name}")
    if metrics["implementation_sha256"] != validation["implementation_sha256"]:
        raise ValueError(f"Training/evaluation implementation mismatch: {directory.name}")
    for field, path in (("implementation_sha256", ROOT / f"{metrics['implementation']}.py"),
                        ("evaluator_sha256", ROOT / "evaluate.py"),
                        ("tokenizer_sha256", ROOT / "data/tokenizer.json")):
        if validation[field] != sha(path):
            raise ValueError(f"Source hash mismatch: {directory.name}: {field}")
    config_path = ROOT / ("configs/baseline.json" if name == "baseline"
                          else f"configs/student_{name}.json")
    if metrics["config_sha256"] != sha(config_path) or plan["config"] != json.loads(config_path.read_text()):
        raise ValueError(f"Config mismatch: {directory.name}")
    if plan["trainer_sha256"] != sha(ROOT / "train_experiment.py"):
        raise ValueError(f"Trainer changed: {directory.name}")
    return dict(run=directory.name, method=name, seed=seed, bpb=validation["bpb"],
                parameters=metrics["parameters"], train_targets=metrics["train_tokens"],
                train_seconds=metrics["train_seconds"], process_seconds=metrics["process_seconds"],
                cpu_seconds_single=validation["seconds"], checkpoint_bytes=metrics["checkpoint_bytes"],
                checkpoint_sha256=checkpoint_hash, gpu_validation_curve=metrics["validation_history"])


def paired_summary(rows: list[dict]) -> dict:
    by_key = {(row["method"], row["seed"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate method/seed")
    summaries = {}
    for method in METHODS:
        scores = [by_key[(method, seed)]["bpb"] for seed in SEEDS]
        deltas = [by_key[(method, seed)]["bpb"] - by_key[("baseline", seed)]["bpb"] for seed in SEEDS]
        summaries[method] = dict(seeds=list(SEEDS), bpb=scores,
                                 mean_bpb=statistics.mean(scores), sample_std_bpb=statistics.stdev(scores),
                                 paired_delta_vs_baseline=deltas,
                                 mean_paired_delta=statistics.mean(deltas),
                                 seeds_better_than_baseline=sum(delta < 0 for delta in deltas))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage2_summary.json")
    args = parser.parse_args()
    rows = [read_run(method, seed) for method in METHODS for seed in SEEDS]
    rows += [read_run(method, 17) for method in LADDER[1:-1]]
    summaries = paired_summary(rows)
    by_key = {(row["method"], row["seed"]): row for row in rows}
    ladder = []
    previous = None
    for method in LADDER:
        bpb = by_key[(method, 17)]["bpb"]
        ladder.append(dict(method=method, seed=17, bpb=bpb,
                           delta_vs_previous=None if previous is None else bpb - previous))
        previous = bpb
    new_rows = [row for row in rows if row["run"].startswith("stage2-")]
    result = dict(split="validation", precision="fp32", device="cpu", summaries=summaries,
                  conditional_ladder=ladder, runs=rows,
                  new_training_targets=sum(row["train_targets"] for row in new_rows),
                  new_train_seconds=sum(row["train_seconds"] for row in new_rows),
                  new_process_seconds=sum(row["process_seconds"] for row in new_rows),
                  caveat="Three seeds are descriptive replication, not a rank guarantee. The ladder has one seed and conditional, order-dependent effects. CPU single-run times are not a repeated resource benchmark.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
