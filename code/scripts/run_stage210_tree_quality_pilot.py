"""Fixed matched-target quality pilot admitted by Stage210 backend preflight."""
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
from train_experiment import checkpoint_payload as original_checkpoint_payload
from train_experiment import learning_rate as original_learning_rate


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage208_lexical_hierarchy_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
IMPLEMENTATION = "student_stage209_tree_propagation"
GRAPH = Path("results/stage210-openvino-tree-a/stage210-random-lexical-head.onnx")


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return original_learning_rate(step, 7200, *args)


def candidate_make_model(implementation: str, config: dict, device):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed implementation")
    return original_make_model(IMPLEMENTATION, config, device)


def candidate_payload(model, implementation: str, config: dict,
                      seed: int, train_tokens: int):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed checkpoint implementation")
    return original_checkpoint_payload(model, IMPLEMENTATION, config, seed,
                                       train_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 trainer changed")
    if (ROOT / CONFIG).read_bytes() != (ROOT / CONTROL).read_bytes():
        raise ValueError("Stage210 changed the matched Stage54 configuration")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (preflight.get("admit_matched_training_pilot") is not True
            or preflight.get("no_data_split_opened") is not True
            or preflight.get("test_scored") is not False
            or preflight.get("precision") != "fp32"
            or preflight.get("model_source_sha256") != sha(ROOT / f"{IMPLEMENTATION}.py")
            or preflight.get("source_sha256") != sha(
                ROOT / "scripts/preflight_stage210_openvino_tree_head.py")
            or preflight.get("graph_sha256") != sha(ROOT / GRAPH)):
        raise ValueError("Fixed Stage210 backend preflight did not pass")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 first 2400 steps, batch32, exact 7200-step LR prefix; "
        "only added jointly trained normalized lexical tree; Stage210 passed "
        "synthetic FP32 head backend gate"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (2400,)
    base.learning_rate = matched_learning_rate
    base.make_model = candidate_make_model
    base.checkpoint_payload = candidate_payload
    base.load_data = load_train_validation
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES if name != CONTROL.as_posix()
    ) + (
        CONFIG.as_posix(), f"{IMPLEMENTATION}.py",
        "student_stage208_lexical_hierarchy.py",
        "scripts/preflight_stage210_openvino_tree_head.py",
        "scripts/run_stage210_tree_quality_pilot.py",
        "scripts/run_stage210_pilot_windows.ps1",
        "scripts/start_stage210_pilot_windows.ps1",
        "scripts/run_stage210_pilot_windows_b.ps1",
        "scripts/start_stage210_pilot_windows_b.ps1",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "docs/STAGE210_OPENVINO_TREE_HEAD_PLAN_20260928.md",
        "tests/test_stage209_tree_propagation.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
