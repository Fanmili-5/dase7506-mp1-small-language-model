"""Bounded, validation-only A/B/C screen; preflight gates precede training.

All outputs are new, under a dedicated run directory. CPU probes score random
initializations solely for resource feasibility, never quality selection.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from common import PROTOCOL, make_model, sha
from train_experiment import atomic_json_dump, checkpoint_payload

SPECS = (
    ("A-wide", "student", "architecture_a_w320_d4.json"),
    ("B-copy", "student_structured", "architecture_b_copy64.json"),
    ("C-mos", "student_structured", "architecture_c_mos2.json"),
)
ASSET_FILES = ("student.py", "student_structured.py", "common.py", "evaluate.py",
               "data/tokenizer.json", "requirements.txt")
SOURCE_FILES = (*ASSET_FILES, "train_experiment.py", "scripts/average_checkpoints.py",
                "scripts/benchmark_cpu.py", "scripts/cpu_resource_probe.py",
                "scripts/peak_memory.py", "scripts/run_architecture_screen.py",
                *(f"configs/{config}" for _, _, config in SPECS))


def execute(arguments, log=None):
    command = [sys.executable, *map(str, arguments)]
    print(json.dumps({"command": command}), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        with Path(log).open("x", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)


def asset_total(checkpoint):
    paths = [checkpoint, *(ROOT / name for name in ASSET_FILES)]
    return sum(path.stat().st_size for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    destination = args.run_dir.resolve()
    if destination.exists():
        parser.error("Choose a new run directory; existing output is never overwritten.")
    if not args.baseline.is_file():
        parser.error("Baseline checkpoint is missing.")
    control = json.loads((ROOT / "results/stage12_summary.json").read_text())
    if control["protocol"] != PROTOCOL or control["split"] != "validation":
        raise ValueError("Unexpected control protocol.")
    if (control["trainer_sha256"] != sha(ROOT / "train_experiment.py")
            or control["planned_steps"] != 7200 or control["train_targets"] != 58982400
            or control["seed"] != 17):
        raise ValueError("The control no longer matches the frozen trainer/budget/seed.")
    endpoint_control = next(row["validation_bpb"] for row in control["candidates"]
                            if row["checkpoint"] == "checkpoint.pt")
    average_control = control["selected"]["validation_bpb"]
    if control["selected"]["checkpoint"] != "average-last5.pt":
        raise ValueError("Unexpected averaging control.")
    destination.mkdir(parents=True)
    (destination / "preflight").mkdir()
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    state = {
        "protocol": PROTOCOL, "split": "validation", "seed": 17,
        "steps": 7200, "targets_per_candidate": 58982400,
        "maximum_training_targets": 176947200,
        "source_hashes": sources, "baseline_sha256": sha(args.baseline),
        "control_summary_sha256": sha(ROOT / "results/stage12_summary.json"),
        "endpoint_control_bpb": endpoint_control, "average_control_bpb": average_control,
        "status": "preflight", "candidates": [],
        "note": "Random-init probes are resource-only. No test evaluation or seed search.",
    }

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json_dump(state, destination / "screen.json")

    def verify_sources():
        for name, expected in sources.items():
            if sha(ROOT / name) != expected:
                raise RuntimeError(f"Source changed during screen: {name}")
        if sha(args.baseline) != state["baseline_sha256"]:
            raise RuntimeError("Baseline checkpoint changed during screen")

    save()
    try:
        execute(["scripts/verify_fixed_files.py"])
        for name, implementation, config_name in SPECS:
            verify_sources()
            config_path = ROOT / "configs" / config_name
            config = json.loads(config_path.read_text())
            torch.manual_seed(17)
            model, _ = make_model(implementation, config, torch.device("cpu"))
            probe = destination / "preflight" / f"{name}.pt"
            parameters = sum(p.numel() for p in model.parameters())
            torch.save(checkpoint_payload(model, implementation, config, 17, 0), probe)
            del model
            gc.collect()
            resource = destination / "preflight" / f"{name}-resource.json"
            execute(["scripts/benchmark_cpu.py", "--baseline", args.baseline,
                     "--candidate", probe, "--repeats", 3, "--threads", 4,
                     "--output", resource], resource.with_suffix(".log"))
            measured = json.loads(resource.read_text())
            assets = asset_total(probe)
            passed = (measured["within_five_x_time_limit"]
                      and measured["within_four_gib_peak_rss_limit"]
                      and assets <= 64 * 1024**2)
            row = {"name": name, "implementation": implementation, "config": config_name,
                   "parameters": parameters, "preflight_pass": passed,
                   "preflight_resource": str(resource), "preflight_asset_bytes": assets,
                   "preflight_time_ratio": measured["candidate_to_baseline_time_ratio"],
                   "status": "eligible" if passed else "resource_rejected"}
            state["candidates"].append(row)
            print(json.dumps(row), flush=True)
            save()

        state["status"] = "training"
        save()
        for row in state["candidates"]:
            if not row["preflight_pass"]:
                continue
            verify_sources()
            name = row["name"]
            directory = destination / name
            row["status"] = "training"
            save()
            arguments = ["train_experiment.py", "--implementation", row["implementation"],
                         "--config", ROOT / "configs" / row["config"],
                         "--device", "cuda", "--precision", "auto", "--seed", 17,
                         "--steps", 7200, "--micro-batch-size", 32, "--grad-accum", 1,
                         "--schedule", "baseline", "--eval-every", 300,
                         "--save-every", 200, "--keep-eval-checkpoints", "--run-dir", directory]
            # Two real updates are an early CUDA smoke check, not an extra run.
            # Resume preserves the exact planned schedule, optimizer, and RNG.
            execute([*arguments, "--stop-after-step", 2])
            execute([*arguments, "--resume"])
            verify_sources()
            metrics = json.loads((directory / "metrics.json").read_text())
            if metrics["train_tokens"] != 58982400 or metrics["seed"] != 17:
                raise ValueError("Training budget/seed mismatch")
            row["train_targets"] = metrics["train_tokens"]
            row["train_seconds"] = metrics["train_seconds"]
            averaging = ["scripts/average_checkpoints.py"]
            for step in (6000, 6300, 6600, 6900, 7200):
                averaging.extend(["--checkpoint", directory / "checkpoints" / f"step-{step:06d}.pt"])
            average = directory / "average-last5.pt"
            execute([*averaging, "--output", average])
            row["status"] = "cpu_validation"
            save()
            for checkpoint_name, label in (("checkpoint.pt", "endpoint"), ("average-last5.pt", "average")):
                checkpoint = directory / checkpoint_name
                output = directory / f"{label}-validation-cpu-fp32.json"
                execute(["evaluate.py", "--checkpoint", checkpoint, "--device", "cpu",
                         "--precision", "fp32", "--threads", 4, "--split", "validation",
                         "--output", output])
                evaluated = json.loads(output.read_text())
                if evaluated["checkpoint_sha256"] != sha(checkpoint) or evaluated["targets"] != 376599:
                    raise ValueError("Validation checkpoint/coverage mismatch")
                row[f"{label}_bpb"] = evaluated["bpb"]
                row[f"{label}_checkpoint_sha256"] = evaluated["checkpoint_sha256"]
            row["endpoint_gain"] = endpoint_control - row["endpoint_bpb"]
            row["average_gain"] = average_control - row["average_bpb"]
            row["final_asset_bytes"] = asset_total(average)
            # Only quality-positive models need a final three-repeat resource gate.
            if row["average_gain"] >= 0.003:
                row["status"] = "final_resource_gate"
                save()
                resource = directory / "final-resource.json"
                execute(["scripts/benchmark_cpu.py", "--baseline", args.baseline,
                         "--candidate", average, "--repeats", 3, "--threads", 4,
                         "--output", resource], resource.with_suffix(".log"))
                measured = json.loads(resource.read_text())
                row["final_time_ratio"] = measured["candidate_to_baseline_time_ratio"]
                row["final_peak_rss_bytes"] = measured["candidate"]["max_peak_rss_bytes"]
                row["final_resource_pass"] = (measured["within_five_x_time_limit"]
                                              and measured["within_four_gib_peak_rss_limit"]
                                              and row["final_asset_bytes"] <= 64 * 1024**2)
            row["status"] = "completed"
            print(json.dumps(row), flush=True)
            save()
        verify_sources()
        execute(["scripts/verify_fixed_files.py"])
        state["status"] = "completed"
        state["selection_note"] = "Screen only; paired replication and mechanism ablation still required."
        save()
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
