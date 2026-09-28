"""Fixed Stage54-matched tied-depth pilot after separate course-aligned admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model as original_make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import (checkpoint_payload as original_checkpoint_payload,
                              learning_rate as original_learning_rate)


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage221_tied_depth_rdrop.json")
CONTROL = Path("configs/stage150_depth10_hybrid_rdrop.json")
IMPLEMENTATION = "student_stage221_tied_depth_rdrop"
FIXED_GRAPH_SHA = "2116a21aff879a8bbd172b1ff5fd43a258a210e32d7eb779d160319ef0774ea2"
FIXED_STAGE143_GRAPH_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
MAX_PROJECTED_RATIO = 4.5
FEATURE_CALLS = 46


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return original_learning_rate(step, 7200, *args)


def candidate_make_model(implementation: str, config: dict, device):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed implementation")
    return original_make_model(IMPLEMENTATION, config, device)


def candidate_checkpoint_payload(model, implementation: str, *args):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed checkpoint implementation")
    return original_checkpoint_payload(model, IMPLEMENTATION, *args)


def validate_course_aligned_admission(preflight: dict, frozen: dict) -> float:
    """Admit quality pilot only; this is not actual resource qualification."""
    required_hashes = {
        "config_sha256": sha(ROOT / CONFIG),
        "source_sha256": sha(ROOT / "scripts/preflight_stage221_tied_depth.py"),
        "training_model_sha256": sha(ROOT / "student_stage221_tied_depth_rdrop.py"),
        "inference_model_sha256": sha(ROOT / "student_stage221_tied_depth_structured.py"),
        "core_model_sha256": sha(ROOT / "student_stage221_tied_depth.py"),
        "candidate_graph_sha256": FIXED_GRAPH_SHA,
        "reference_graph_sha256": FIXED_STAGE143_GRAPH_SHA,
    }
    if any(preflight.get(key) != value for key, value in required_hashes.items()):
        raise ValueError("Stage221 architecture/preflight binding changed")
    if (preflight.get("status") != "synthetic_input_only_preflight"
            or preflight.get("no_validation_or_test_scoring") is not True
            or preflight.get("admit_matched_training_pilot") is not False
            or frozen.get("qualified") is not True
            or frozen.get("no_test_scoring") is not True
            or frozen.get("graph_sha256") != FIXED_STAGE143_GRAPH_SHA):
        raise ValueError("Expected failed Stage221 and qualified Stage143 evidence")
    if (preflight["projected_conservative_assets_bytes"] > 64 * 1024 * 1024
            or preflight["openvino_hidden_max_error"] > 3e-4
            or preflight["max_future_prefix_error"] > 1e-5
            or preflight["max_independent_row_error"] > 1e-5
            or preflight["synthetic_optimizer_steps"] != 2
            or preflight["gpu_peak_reserved_bytes"] >= preflight["gpu_total_bytes"]):
        raise ValueError("Stage221 non-time preflight failed")
    feature_delta = (preflight["median_seconds"]["candidate"]
                     - preflight["median_seconds"]["reference"])
    projected_seconds = frozen["candidate_median_seconds"] + FEATURE_CALLS * feature_delta
    ratio = projected_seconds / frozen["baseline_median_seconds"]
    if feature_delta < 0 or ratio > MAX_PROJECTED_RATIO:
        raise ValueError(f"Course-aligned projected CPU ratio {ratio:.6f} > 4.5")
    print(json.dumps(dict(stage="stage222_quality_pilot_admission",
                          stage221_passed=False, projected_total_cpu_ratio=ratio,
                          projected_total_cpu_seconds=projected_seconds,
                          actual_resource_qualified=False)), flush=True)
    return ratio


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--frozen-resource", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 trainer changed")
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    control = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control.keys()
               if config.get(key) != control.get(key)}
    if changed != {"shared_tail_pair"} or config["shared_tail_pair"] != [7, 8]:
        raise ValueError("Fixed architecture changed")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    frozen = json.loads(args.frozen_resource.read_text(encoding="utf-8-sig"))
    validate_course_aligned_admission(preflight, frozen)
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage222 course-aligned re-admission of exact Stage221 tied-depth model; "
        "Stage54 seed-17 first 2400 steps, batch32, exact 7200-step LR prefix"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (2400,)
    base.learning_rate = matched_learning_rate
    base.make_model = candidate_make_model
    base.checkpoint_payload = candidate_checkpoint_payload
    base.load_data = load_train_validation
    base.SOURCE_FILES = base.SOURCE_FILES + (
        CONFIG.as_posix(), CONTROL.as_posix(),
        "student_stage221_tied_depth.py",
        "student_stage221_tied_depth_rdrop.py",
        "student_stage221_tied_depth_structured.py",
        "scripts/preflight_stage221_tied_depth.py",
        "scripts/run_stage222_course_aligned_tied_pilot.py",
        "scripts/run_stage222_pilot_windows.ps1",
        "scripts/start_stage222_pilot_windows.ps1",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "docs/STAGE221_TIED_DEPTH_PLAN_20260928.md",
        "docs/STAGE221_PREFLIGHT_RESULT_20260928.md",
        "docs/STAGE222_COURSE_ALIGNED_TIED_PILOT_PLAN_20260928.md",
        "tests/test_stage221_tied_depth.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
