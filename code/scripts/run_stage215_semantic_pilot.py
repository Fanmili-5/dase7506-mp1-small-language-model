"""Fixed 2,400-step Stage54-matched train/validation-only Stage215 pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model as original_make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.build_stage215_semantic_basis import build_basis
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import (checkpoint_payload as original_checkpoint_payload,
                              learning_rate as original_learning_rate)


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage215_semantic_input_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
IMPLEMENTATION = "student_stage215_semantic_input"
EXPECTED_BASIS_SHA = ""


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return original_learning_rate(step, 7200, *args)


def candidate_make_model(implementation: str, config: dict, device):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed implementation")
    model, digest = original_make_model(IMPLEMENTATION, config, device)
    basis, metadata = build_basis()
    if metadata["basis_sha256"] != EXPECTED_BASIS_SHA:
        raise ValueError("Train-derived semantic basis differs from Windows preflight")
    model.set_semantic_basis(basis.to(device))
    return model, digest


def candidate_checkpoint_payload(model, implementation: str, *args):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed checkpoint implementation")
    return original_checkpoint_payload(model, IMPLEMENTATION, *args)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 trainer changed")
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    control = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control.keys()
               if config.get(key) != control.get(key)}
    if changed != {"semantic_dim"} or config["semantic_dim"] != 64:
        raise ValueError("Stage215 changed outside the fixed semantic input")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (preflight.get("admit_quality_pilot") is not True
            or preflight.get("no_validation_or_test_scoring") is not True
            or preflight.get("config_sha256") != sha(ROOT / CONFIG)
            or preflight.get("source_sha256")
            != sha(ROOT / "scripts/preflight_stage215_semantic_input.py")
            or preflight.get("implementation_sha256")
            != sha(ROOT / "student_stage215_semantic_input.py")):
        raise ValueError("Fixed Stage215 input-only preflight did not pass")
    global EXPECTED_BASIS_SHA
    EXPECTED_BASIS_SHA = preflight["basis"]["basis_sha256"]
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 first 2400 steps, batch32, exact 7200-step LR prefix; "
        "only a fixed train-derived rank64 semantic input stream is added"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (2400,)
    base.learning_rate = matched_learning_rate
    base.make_model = candidate_make_model
    base.checkpoint_payload = candidate_checkpoint_payload
    base.load_data = load_train_validation
    base.SOURCE_FILES = base.SOURCE_FILES + (
        CONFIG.as_posix(),
        "student_stage215_semantic_input.py",
        "scripts/build_stage215_semantic_basis.py",
        "scripts/preflight_stage215_semantic_input.py",
        "scripts/run_stage215_semantic_pilot.py",
        "scripts/run_stage215_pilot_windows.ps1",
        "scripts/start_stage215_pilot_windows.ps1",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "docs/STAGE215_TRAIN_SEMANTIC_INPUT_PLAN_20260928.md",
        "tests/test_stage215_semantic_input.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
