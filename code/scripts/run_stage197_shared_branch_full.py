"""Train the existing Stage153 shared-branch architecture for 7,200 steps.

Overrides the historical Stage54 data loader so the test file is not opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import checkpoint_payload

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage153_shared_branch_rdrop.json")
IMPLEMENTATION = "student_stage153_shared_branch"


def stage197_model(_implementation, config, device):
    return make_model(IMPLEMENTATION, config, device)


def stage197_checkpoint_payload(model, _implementation, config, seed, tokens):
    return checkpoint_payload(model, IMPLEMENTATION, config, seed, tokens)


def main() -> None:
    preflight_parser = argparse.ArgumentParser(add_help=False)
    preflight_parser.add_argument("--preflight", type=Path, required=True)
    args, remaining = preflight_parser.parse_known_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (preflight.get("implementation") != IMPLEMENTATION
            or not preflight.get("training_gate_passed")
            or preflight.get("config_sha256") != sha(ROOT / CONFIG)):
        raise ValueError("Stage197 GPU preflight did not pass")
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 trainer changed")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 full 7,200-step R-Drop control with matched "
        "training targets; six shared blocks and two heterogeneous upper paths"
    )
    base.make_model = stage197_model
    base.checkpoint_payload = stage197_checkpoint_payload
    base.load_data = load_train_validation
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != "configs/stage54_hybrid_conv_rdrop.json"
    ) + (
        CONFIG.as_posix(), "student_stage153_shared_branch.py",
        "scripts/run_stage197_shared_branch_full.py",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
