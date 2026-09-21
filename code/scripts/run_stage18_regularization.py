"""Two isolated, equal-target regularizers; identical exported inference graph."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, sha
from student_regularized import inference_config
from train_experiment import atomic_json_dump
from scripts.run_architecture_screen import execute, asset_total
from scripts.run_stage15_screen import average_steps

SPECS = (("H-embedding-drop", "stage18_embedding_drop.json"),
         ("I-hidden-drop", "stage18_hidden_drop.json"))
STEPS, TARGETS = 7200, 58982400
REFERENCE_SHA = "5783e16de954168793148d3a95c7559ddd6ec2916987247ea060cd459801aa16"
SOURCES = ("student.py", "student_structured.py", "student_regularized.py", "train_experiment.py",
    "scripts/run_stage18_regularization.py", "scripts/export_regularized.py", "tests/test_regularized.py",
    "scripts/average_checkpoints.py", "scripts/benchmark_cpu.py", "scripts/cpu_resource_probe.py",
    "scripts/peak_memory.py", "scripts/run_architecture_screen.py", "scripts/run_stage15_screen.py",
    "scripts/verify_fixed_files.py", "common.py", "evaluate.py", "data/tokenizer.json",
    "data/manifest.json", "requirements.txt", "results/stage15-audit.json", "configs/stage15_copy_depth8.json",
    *(f"configs/{config}" for _, config in SPECS))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--stage17-screen", type=Path, required=True)
    args = p.parse_args()
    if args.run_dir.exists():
        raise FileExistsError("Choose a new run directory")
    previous = json.loads(args.stage17_screen.read_text())
    if previous["status"] != "completed_teacher_rejected":
        raise ValueError("Expected the completed, rejected teacher screen; do not overlap GPU jobs")
    if sha(args.reference) != REFERENCE_SHA or sha(args.baseline) != previous["baseline_sha256"]:
        raise ValueError("Reference/baseline checkpoint mismatch")
    audit = json.loads((ROOT / "results/stage15-audit.json").read_text())
    reference = next(r for r in audit["candidates"] if r["name"] == "F-copy-depth8")
    if not reference["eligible_for_followup"] or reference["average_checkpoint_sha256"] != REFERENCE_SHA:
        raise ValueError("Missing qualified F reference")
    base_config = json.loads((ROOT / "configs/stage15_copy_depth8.json").read_text())
    for _, name in SPECS:
        if inference_config(json.loads((ROOT / "configs" / name).read_text())) != base_config:
            raise ValueError("Exported model must be identical in architecture to F")
    state = dict(protocol=PROTOCOL, split="validation", seed=17, status="preflight", candidates=[],
        source_hashes={name: sha(ROOT / name) for name in SOURCES},
        previous_screen_sha256=sha(args.stage17_screen), baseline_sha256=sha(args.baseline),
        reference_checkpoint_sha256=REFERENCE_SHA, reference_bpb=reference["average_bpb"],
        maximum_new_training_targets=2 * TARGETS,
        selection="Fixed last5 6000..7200 every300, >=.003 BPB gain triggers final resources; no test",
        note="Same trainer, seed, initial parameters, sampling and 7200 updates as F. Dropout RNG thereafter differs.")
    args.run_dir.mkdir(parents=True)

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json_dump(state, args.run_dir / "screen.json")

    def verify():
        for name, digest in state["source_hashes"].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Changed pinned source: {name}")
        if sha(args.reference) != REFERENCE_SHA or sha(args.baseline) != state["baseline_sha256"]:
            raise ValueError("Changed reference checkpoint")
        if sha(args.stage17_screen) != state["previous_screen_sha256"]:
            raise ValueError("Changed previous experiment receipt")

    save()
    try:
        execute(["scripts/verify_fixed_files.py"])
        execute(["-m", "unittest", "discover", "-s", "tests", "-p", "test_regularized.py", "-v"])
        for name, config in SPECS:
            verify()
            run = args.run_dir / name
            row = dict(name=name, config=config, planned_targets=TARGETS, status="smoke")
            state["candidates"].append(row)
            state["status"] = "training"
            save()
            command = ["train_experiment.py", "--implementation", "student_regularized",
                "--config", ROOT / "configs" / config, "--run-dir", run, "--device", "cuda",
                "--precision", "bf16", "--threads", 4, "--seed", 17, "--steps", STEPS,
                "--micro-batch-size", 32, "--grad-accum", 1, "--schedule", "baseline",
                "--eval-every", 300, "--eval-batch-size", 32, "--save-every", 200, "--keep-eval-checkpoints"]
            execute([*command, "--stop-after-step", 2])
            row["status"] = "training"
            save()
            execute([*command, "--resume"])
            verify()
            metrics = json.loads((run / "metrics.json").read_text())
            if metrics["train_tokens"] != TARGETS or metrics["seed"] != 17:
                raise ValueError("Unexpected training budget/seed")
            row.update(train_targets=TARGETS, train_seconds=metrics["train_seconds"])
            average = run / "average-training-last5.pt"
            command = ["scripts/average_checkpoints.py"]
            for step in average_steps(STEPS):
                command.extend(["--checkpoint", run / "checkpoints" / f"step-{step:06d}.pt"])
            execute([*command, "--output", average])
            exported = run / "average-inference.pt"
            execute(["scripts/export_regularized.py", "--checkpoint", average, "--output", exported])
            row["status"] = "cpu_validation"
            save()
            output = run / "average-validation-cpu-fp32.json"
            execute(["evaluate.py", "--checkpoint", exported, "--device", "cpu", "--precision", "fp32",
                     "--threads", 4, "--split", "validation", "--output", output])
            score = json.loads(output.read_text())
            if score["checkpoint_sha256"] != sha(exported) or score["targets"] != 376599:
                raise ValueError("Scoring checkpoint/coverage mismatch")
            row.update(average_bpb=score["bpb"], checkpoint_sha256=sha(exported),
                       gain_over_F=state["reference_bpb"] - score["bpb"],
                       asset_bytes=asset_total(exported), eligible_for_followup=False)
            if row["gain_over_F"] >= .003:
                row["status"] = "final_resource_gate"
                save()
                resource = run / "final-resource.json"
                execute(["scripts/benchmark_cpu.py", "--baseline", args.baseline, "--candidate", exported,
                         "--repeats", 3, "--threads", 4, "--output", resource], resource.with_suffix(".log"))
                measured = json.loads(resource.read_text())
                row["final_time_ratio"] = measured["candidate_to_baseline_time_ratio"]
                row["eligible_for_followup"] = (measured["within_five_x_time_limit"] and
                    measured["within_four_gib_peak_rss_limit"] and row["asset_bytes"] <= 64 * 1024 ** 2)
            row["status"] = "completed"
            save()
        verify()
        state["actual_new_training_targets"] = sum(r["train_targets"] for r in state["candidates"])
        state["status"] = "completed"
        save()
    except BaseException as error:
        state["status"], state["error"] = "failed", str(error)
        save()
        raise


if __name__ == "__main__":
    main()
