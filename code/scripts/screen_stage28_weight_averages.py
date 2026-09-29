"""Validation-only screen of same-trajectory averages and one compatible weight soup."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score
from scripts.average_checkpoints import average_checkpoints
from student_deep_supervision import inference_config, inference_state
from train_experiment import atomic_json_dump, atomic_torch_save

STAGE22_SHA = "05f20e68313a3c41c4a1116bc3d335ad688507c933fb60e9176a83d8953e5269"
H_SHA = "2be8f4f4ac195593835aaa74f042b0b5d0f5473a46eb6d09fba99b3c11d263f7"
SUBSETS = {
    "last2": (6900, 7200),
    "last3": (6600, 6900, 7200),
    "last4": (6300, 6600, 6900, 7200),
    "last5": (6000, 6300, 6600, 6900, 7200),
}
SOUP_STAGE22_WEIGHTS = (.50, .75, .875)


def export_deep_payload(training_payload, name, source_paths):
    config = inference_config(training_payload["config"])
    deployed, _ = make_model("student_structured", config, torch.device("cpu"))
    deployed.load_state_dict(inference_state(training_payload["model"]), strict=True)
    return dict(
        training_payload, implementation="student_structured", config=config,
        model=deployed.state_dict(),
        training_deep_supervision=dict(
            selection_name=name, source_checkpoint_sha256=[sha(path) for path in source_paths],
            layers=training_payload["config"]["deep_supervision_layers"],
            weight=training_payload["config"]["deep_supervision_weight"],
            same_unique_next_token_targets=True,
            note="Training-only auxiliary norms removed after same-trajectory averaging.",
        ),
    )


def soup_payload(stage22, regularized, stage22_weight):
    if (stage22["protocol"] != PROTOCOL or regularized["protocol"] != PROTOCOL
            or stage22["implementation"] != "student_structured"
            or regularized["implementation"] != "student_structured"
            or stage22["config"] != regularized["config"]
            or stage22["model"].keys() != regularized["model"].keys()):
        raise ValueError("Weight-soup sources are incompatible")
    state = {}
    for key, left in stage22["model"].items():
        right = regularized["model"][key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("Weight-soup tensor metadata mismatch")
        if left.is_floating_point():
            state[key] = (left.double() * stage22_weight + right.double() * (1 - stage22_weight)).to(left.dtype)
        else:
            if not torch.equal(left, right):
                raise ValueError("Non-floating weight-soup state differs")
            state[key] = left.clone()
    return dict(
        stage22, model=state,
        weight_soup=dict(stage22_weight=stage22_weight, stage22_sha256=STAGE22_SHA,
                         stage18_h_sha256=H_SHA, new_gradient_targets=0,
                         note="Compatible same-seed weight interpolation selected on validation only."),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage22-dir", type=Path, required=True)
    parser.add_argument("--stage18-h", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    stage22_path = args.stage22_dir / "average-inference.pt"
    if sha(stage22_path) != STAGE22_SHA or sha(args.stage18_h) != H_SHA:
        raise ValueError("Fixed source checkpoint changed")
    stage22 = torch.load(stage22_path, map_location="cpu", weights_only=True)
    regularized = torch.load(args.stage18_h, map_location="cpu", weights_only=True)
    args.run_dir.mkdir(parents=True)
    candidates = []
    for name, steps in SUBSETS.items():
        paths = [args.stage22_dir / "checkpoints" / f"step-{step:06d}.pt" for step in steps]
        training = average_checkpoints(paths)
        payload = export_deep_payload(training, name, paths)
        path = args.run_dir / f"{name}.pt"
        atomic_torch_save(payload, path)
        candidates.append((name, path, dict(kind="same_trajectory_average", steps=list(steps))))
    for weight in SOUP_STAGE22_WEIGHTS:
        name = f"soup-stage22-{weight:.3f}"
        path = args.run_dir / f"{name}.pt"
        atomic_torch_save(soup_payload(stage22, regularized, weight), path)
        candidates.append((name, path, dict(kind="stage22_stage18h_weight_soup", stage22_weight=weight)))
    device, _ = setup("cuda", "fp32", 4)
    data = load_data()
    rows = []
    started = time.perf_counter()
    for name, path, recipe in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model, _ = make_model(payload["implementation"], payload["config"], device)
        model.load_state_dict(payload["model"])
        result = score(model, *data["validation"], device, "fp32", 32)
        result.pop("window_nll_nats")
        row = dict(name=name, checkpoint_sha256=sha(path), checkpoint_bytes=path.stat().st_size,
                   recipe=recipe, **result)
        rows.append(row); print(json.dumps(row), flush=True)
        del model
    best = min(rows, key=lambda row: row["bpb"])
    best_path = args.run_dir / f"{best['name']}.pt"
    official_path = args.run_dir / "best-validation-cpu-fp32.json"
    subprocess.run([sys.executable, "evaluate.py", "--checkpoint", str(best_path.resolve()),
                    "--device", "cpu", "--precision", "fp32", "--threads", "4",
                    "--split", "validation", "--output", str(official_path.resolve())],
                   cwd=ROOT, check=True)
    official = json.loads(official_path.read_text(encoding="utf-8"))
    if abs(official["bpb"] - best["bpb"]) > 1e-5:
        raise ValueError("GPU screen and CPU verification disagree")
    report = dict(protocol=PROTOCOL, split="validation", candidates=rows, best=best,
                  best_checkpoint=str(best_path), official_cpu_bpb=official["bpb"],
                  seconds=time.perf_counter() - started, new_gradient_targets=0,
                  note="Small predeclared averaging/soup grid; validation-only selection and no test scoring.")
    atomic_json_dump(report, args.run_dir / "screen.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
