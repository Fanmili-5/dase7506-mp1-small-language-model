"""Bounded Stage 15: matched ablation, successor route, capacity and duration.

All models start from scratch. Validation only; no automatic budget expansion.
The long run is explicitly NOT an equal-training-target architecture comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from common import PROTOCOL, make_model, sha
from train_experiment import atomic_json_dump, checkpoint_payload
from scripts.run_architecture_screen import execute

SPECS = (
    ("D-fp32-control", "student_successor", "stage15_fp32_control.json", 7200),
    ("E-dual-copy", "student_successor", "stage15_dual_copy.json", 7200),
    ("F-copy-depth8", "student_structured", "stage15_copy_depth8.json", 7200),
    ("G-copy-long", "student_structured", "architecture_b_copy64.json", 21600),
)
ASSET_FILES = ("student.py", "student_structured.py", "student_successor.py", "common.py",
               "evaluate.py", "data/tokenizer.json", "requirements.txt")
SOURCE_FILES = (*ASSET_FILES, "train_experiment.py", "scripts/average_checkpoints.py",
                "scripts/benchmark_cpu.py", "scripts/cpu_resource_probe.py",
                "scripts/peak_memory.py", "scripts/run_architecture_screen.py",
                "scripts/run_stage15_screen.py", "scripts/verify_fixed_files.py",
                "results/stage14-audit.json", "results/stage14-evidence/B-copy/run.json",
                *(f"configs/{name}" for _, _, name, _ in SPECS))
REFERENCE_SHA = "966dcc405ba3d084d06912bdf8136da561e0473dfd119b1ed9f07d223dd9ed0b"


def average_steps(steps):
    if steps < 1500 or steps % 300:
        raise ValueError("Plan requires five final 300-update snapshots")
    return tuple(range(steps - 1200, steps + 1, 300))


def asset_total(checkpoint):
    return sum(path.stat().st_size for path in [checkpoint, *(ROOT / x for x in ASSET_FILES)])


def resource_pass(result, assets):
    return bool(result["within_five_x_time_limit"] and result["within_four_gib_peak_rss_limit"]
                and assets <= 64 * 1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if directory.exists():
        parser.error("Use a new run directory")
    if not args.baseline.is_file() or sha(args.reference) != REFERENCE_SHA:
        raise ValueError("Missing baseline or unexpected B reference checkpoint")
    audit = json.loads((ROOT / "results/stage14-audit.json").read_text())
    reference = next(r for r in audit["candidates"] if r["name"] == "B-copy")
    if (audit["protocol"] != PROTOCOL or audit["split"] != "validation"
            or reference["average_checkpoint_sha256"] != REFERENCE_SHA):
        raise ValueError("Unexpected B evidence")
    previous = json.loads((ROOT / "results/stage14-evidence/B-copy/run.json").read_text())
    if (previous["trainer_sha256"] != sha(ROOT / "train_experiment.py")
            or previous["seed"] != 17 or previous["planned_train_targets"] != 58982400):
        raise ValueError("Historical comparison does not match trainer/seed/budget")
    directory.mkdir(parents=True)
    (directory / "preflight").mkdir()
    torch.set_num_threads(4)
    state = {
        "protocol": PROTOCOL, "split": "validation", "seed": 17,
        "status": "preflight", "candidates": [],
        "source_hashes": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "baseline_sha256": sha(args.baseline), "reference_sha256": sha(args.reference),
        "reference_average_bpb": reference["average_bpb"],
        "reference_endpoint_bpb": reference["endpoint_bpb"],
        "maximum_new_training_targets": sum(s[3] for s in SPECS) * 8192,
        "selection_rule": "Compare fixed last5 average; >=0.003 BPB gain triggers exact resource gate. No test.",
        "note": "D/E/F equal 7200 updates; G uses 3x targets and is a duration experiment, not a matched architecture win.",
    }

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json_dump(state, directory / "screen.json")

    def verify_sources():
        for name, expected in state["source_hashes"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError(f"Changed source: {name}")
        if sha(args.reference) != state["reference_sha256"] or sha(args.baseline) != state["baseline_sha256"]:
            raise RuntimeError("Changed reference checkpoint")

    save()
    try:
        execute(["scripts/verify_fixed_files.py"])
        for name, implementation, config_name, steps in SPECS:
            verify_sources()
            config = json.loads((ROOT / "configs" / config_name).read_text())
            row = dict(name=name, implementation=implementation, config=config_name,
                       steps=steps, planned_targets=steps * 8192, average_steps=average_steps(steps))
            state["candidates"].append(row)
            torch.manual_seed(17)
            model, _ = make_model(implementation, config, torch.device("cpu"))
            row["parameters"] = sum(p.numel() for p in model.parameters())
            probe = directory / "preflight" / f"{name}.pt"
            torch.save(checkpoint_payload(model, implementation, config, 17, 0), probe)
            del model
            gc.collect()
            resource = directory / "preflight" / f"{name}-resource.json"
            row["status"] = "resource_preflight"
            save()
            execute(["scripts/benchmark_cpu.py", "--baseline", args.baseline, "--candidate", probe,
                     "--repeats", 3, "--threads", 4, "--output", resource], resource.with_suffix(".log"))
            measured = json.loads(resource.read_text())
            row["preflight_asset_bytes"] = asset_total(probe)
            row["preflight_pass"] = resource_pass(measured, row["preflight_asset_bytes"])
            row["preflight_time_ratio"] = measured["candidate_to_baseline_time_ratio"]
            row["status"] = "eligible" if row["preflight_pass"] else "resource_rejected"
            save()

        state["status"] = "training"
        save()
        for row in state["candidates"]:
            if not row["preflight_pass"]:
                continue
            verify_sources()
            run = directory / row["name"]
            row["status"] = "training"
            save()
            command = ["train_experiment.py", "--implementation", row["implementation"],
                       "--config", ROOT / "configs" / row["config"], "--device", "cuda",
                       "--precision", "auto", "--seed", 17, "--steps", row["steps"],
                       "--micro-batch-size", 32, "--grad-accum", 1, "--schedule", "baseline",
                       "--eval-every", 300, "--save-every", 200, "--keep-eval-checkpoints", "--run-dir", run]
            execute([*command, "--stop-after-step", 2])
            execute([*command, "--resume"])
            verify_sources()
            metrics = json.loads((run / "metrics.json").read_text())
            if metrics["train_tokens"] != row["planned_targets"] or metrics["seed"] != 17:
                raise RuntimeError("Training budget or seed mismatch")
            row["train_targets"], row["train_seconds"] = metrics["train_tokens"], metrics["train_seconds"]
            command = ["scripts/average_checkpoints.py"]
            for step in row["average_steps"]:
                command.extend(["--checkpoint", run / "checkpoints" / f"step-{step:06d}.pt"])
            average = run / "average-last5.pt"
            execute([*command, "--output", average])
            row["status"] = "cpu_validation"
            save()
            for filename, label in (("checkpoint.pt", "endpoint"), ("average-last5.pt", "average")):
                checkpoint = run / filename
                output = run / f"{label}-validation-cpu-fp32.json"
                execute(["evaluate.py", "--checkpoint", checkpoint, "--device", "cpu", "--precision", "fp32",
                         "--threads", 4, "--split", "validation", "--output", output])
                result = json.loads(output.read_text())
                if result["checkpoint_sha256"] != sha(checkpoint) or result["targets"] != 376599:
                    raise RuntimeError("Scoring coverage/hash mismatch")
                row[f"{label}_bpb"] = result["bpb"]
                row[f"{label}_checkpoint_sha256"] = result["checkpoint_sha256"]
                row[f"{label}_gain_over_B"] = state[f"reference_{label}_bpb"] - result["bpb"]
            row["final_asset_bytes"] = asset_total(average)
            if row["average_gain_over_B"] >= 0.003:
                row["status"] = "final_resource_gate"
                save()
                resource = run / "final-resource.json"
                execute(["scripts/benchmark_cpu.py", "--baseline", args.baseline, "--candidate", average,
                         "--repeats", 3, "--threads", 4, "--output", resource], resource.with_suffix(".log"))
                measured = json.loads(resource.read_text())
                row["final_resource_pass"] = resource_pass(measured, row["final_asset_bytes"])
                row["final_time_ratio"] = measured["candidate_to_baseline_time_ratio"]
                row["final_peak_rss_bytes"] = measured["candidate"]["max_peak_rss_bytes"]
            row["status"] = "completed"
            save()
        verify_sources()
        execute(["scripts/verify_fixed_files.py"])
        state["status"] = "completed"
        state["actual_new_training_targets"] = sum(r.get("train_targets", 0) for r in state["candidates"])
        save()
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
