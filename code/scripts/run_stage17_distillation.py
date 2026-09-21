"""Bounded teacher-first experiment, executable only after Stage15 completes.

No seed/temperature grid, test scoring, budget expansion or inference teacher.
An unsuccessful teacher stops the experiment before student retraining.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, sha
from train_experiment import atomic_json_dump
from scripts.run_architecture_screen import execute, asset_total
from scripts.run_stage15_screen import average_steps

STEPS = 14400
TARGETS = STEPS * 8192
MIN_GAIN = 0.003
SOURCES = (
    "train_experiment.py", "train_distillation.py", "student.py", "student_structured.py",
    "common.py", "evaluate.py", "scripts/run_stage17_distillation.py",
    "scripts/run_architecture_screen.py", "scripts/run_stage15_screen.py",
    "scripts/average_checkpoints.py", "scripts/average_distilled_checkpoints.py",
    "scripts/benchmark_cpu.py", "scripts/cpu_resource_probe.py", "scripts/peak_memory.py",
    "scripts/verify_fixed_files.py", "tests/test_distillation.py",
    "configs/architecture_b_copy64.json", "configs/stage17_teacher_copy384.json",
    "data/tokenizer.json", "data/manifest.json", "requirements.txt",
)


def reference_from_stage15(screen):
    if screen["status"] != "completed" or screen["protocol"] != PROTOCOL or screen["split"] != "validation":
        raise ValueError("Stage15 must complete successfully before starting Stage17")
    qualified = [r for r in screen["candidates"] if r.get("final_resource_pass")]
    return min([screen["reference_average_bpb"], *(r["average_bpb"] for r in qualified)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--stage15-screen", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="Without this flag only print the bounded plan")
    args = p.parse_args()
    previous = json.loads(args.stage15_screen.read_text(encoding="utf-8"))
    threshold = reference_from_stage15(previous)
    if sha(args.baseline) != previous["baseline_sha256"]:
        raise ValueError("Baseline does not match Stage15")
    state = dict(protocol=PROTOCOL, split="validation", seed=17, status="planned", candidates=[],
        source_hashes={name: sha(ROOT / name) for name in SOURCES},
        stage15_screen_sha256=sha(args.stage15_screen), baseline_sha256=sha(args.baseline),
        reference_bpb=threshold, minimum_teacher_gain=MIN_GAIN,
        maximum_new_gradient_targets=3 * TARGETS, maximum_teacher_forward_targets=TARGETS,
        teacher_only="10x384 prefix-copy, from scratch on provided train; never an inference asset",
        student="6x256 prefix-copy, CE control and KD alpha=.5 T=2, each 14400 updates",
        selection="Fixed last5 snapshots 13200..14400 every300; full CPU FP32 validation; no test")
    if not args.execute:
        print(json.dumps(state, indent=2))
        return
    if args.run_dir.exists():
        raise FileExistsError("Use a new experiment directory; recovery is explicit, not automatic restart")
    args.run_dir.mkdir(parents=True)

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json_dump(state, args.run_dir / "screen.json")

    def verify():
        for name, expected in state["source_hashes"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError(f"Changed experiment source: {name}")
        if sha(args.stage15_screen) != state["stage15_screen_sha256"] or sha(args.baseline) != state["baseline_sha256"]:
            raise RuntimeError("Changed reference evidence")

    def train_and_score(name, command, is_student):
        verify()
        directory = args.run_dir / name
        row = dict(name=name, status="smoke", planned_targets=TARGETS)
        state["candidates"].append(row)
        state["status"] = name
        save()
        command = [*command, "--run-dir", directory]
        execute([*command, "--stop-after-step", 2])
        row["status"] = "training"
        save()
        execute([*command, "--resume"])
        verify()
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics["train_tokens"] != TARGETS:
            raise ValueError("Unexpected training target count")
        row["train_targets"] = metrics["train_tokens"]
        row["reported_seconds"] = metrics.get("process_seconds", metrics.get("train_seconds"))
        average_script = "scripts/average_distilled_checkpoints.py" if is_student else "scripts/average_checkpoints.py"
        average_command = [average_script]
        for step in average_steps(STEPS):
            average_command.extend(["--checkpoint", directory / "checkpoints" / f"step-{step:06d}.pt"])
        checkpoint = directory / "average-last5.pt"
        execute([*average_command, "--output", checkpoint])
        output = directory / "average-validation-cpu-fp32.json"
        row["status"] = "cpu_validation"
        save()
        execute(["evaluate.py", "--checkpoint", checkpoint, "--device", "cpu", "--precision", "fp32",
                 "--threads", 4, "--split", "validation", "--output", output])
        result = json.loads(output.read_text())
        if result["checkpoint_sha256"] != sha(checkpoint) or result["targets"] != 376599:
            raise ValueError("Scoring hash/coverage mismatch")
        row.update(average_bpb=result["bpb"], checkpoint_sha256=sha(checkpoint), status="scored")
        save()
        return row, checkpoint, directory

    save()
    try:
        execute(["scripts/verify_fixed_files.py"])
        execute(["-m", "unittest", "discover", "-s", "tests", "-p", "test_distillation.py", "-v"])
        common = ["--device", "cuda", "--precision", "bf16", "--threads", 4, "--seed", 17,
                  "--steps", STEPS, "--micro-batch-size", 16, "--grad-accum", 2,
                  "--eval-every", 300, "--eval-batch-size", 16, "--save-every", 300]
        teacher, teacher_path, teacher_dir = train_and_score("teacher", ["train_experiment.py",
            "--implementation", "student_structured", "--config", "configs/stage17_teacher_copy384.json",
            "--keep-eval-checkpoints", *common], False)
        teacher["submission_eligible"] = False
        teacher["quality_pass"] = teacher["average_bpb"] <= threshold - MIN_GAIN
        if not teacher["quality_pass"]:
            state["status"] = "completed_teacher_rejected"
            state["actual_new_gradient_targets"] = TARGETS
            save()
            return
        for name, alpha in (("student-ce", 0), ("student-kd", .5)):
            command = ["train_distillation.py", "--config", "configs/architecture_b_copy64.json",
                       "--alpha", alpha, "--temperature", 2, *common]
            if alpha:
                command.extend(["--teacher", teacher_path, "--teacher-sha256", teacher["checkpoint_sha256"],
                                "--teacher-metrics", teacher_dir / "metrics.json"])
            row, checkpoint, directory = train_and_score(name, command, True)
            row["improves_reference"] = row["average_bpb"] <= threshold - MIN_GAIN
            if row["improves_reference"]:
                resource = directory / "final-resource.json"
                execute(["scripts/benchmark_cpu.py", "--baseline", args.baseline, "--candidate", checkpoint,
                         "--repeats", 3, "--threads", 4, "--output", resource], resource.with_suffix(".log"))
                measured = json.loads(resource.read_text())
                row["asset_bytes"] = asset_total(checkpoint)
                row["resource_pass"] = (measured["within_five_x_time_limit"] and
                    measured["within_four_gib_peak_rss_limit"] and row["asset_bytes"] <= 64 * 1024 ** 2)
                row["time_ratio"] = measured["candidate_to_baseline_time_ratio"]
            save()
        verify()
        execute(["scripts/verify_fixed_files.py"])
        state["status"] = "completed"
        state["actual_new_gradient_targets"] = 3 * TARGETS
        state["actual_teacher_forward_targets"] = TARGETS
        ce, kd = state["candidates"][1:]
        state["kd_bpb_gain_over_matched_ce"] = ce["average_bpb"] - kd["average_bpb"]
        save()
    except BaseException as error:
        state["status"], state["error"] = "failed", str(error)
        save()
        raise


if __name__ == "__main__":
    main()
