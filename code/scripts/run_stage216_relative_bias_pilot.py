"""Fixed 2,400-step Stage54-matched train/validation-only Stage216 pilot."""
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
CONFIG = Path("configs/stage216_relative_bias_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
IMPLEMENTATION = "student_stage216_relative_bias"


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
    if changed != {"relative_bias_buckets"} or config["relative_bias_buckets"] != 32:
        raise ValueError("Stage216 changed outside the fixed relative bias")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (preflight.get("admit_matched_training_pilot") is not True
            or preflight.get("no_validation_or_test_scoring") is not True
            or preflight.get("config_sha256") != sha(ROOT / CONFIG)
            or preflight.get("source_sha256")
            != sha(ROOT / "scripts/preflight_stage216_relative_bias.py")
            or preflight.get("model_source_sha256")
            != sha(ROOT / "student_stage216_relative_bias.py")):
        raise ValueError("Fixed Stage216 input-only preflight did not pass")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 first 2400 steps, batch32, exact 7200-step LR prefix; "
        "only four global-attention blocks add zero-start headwise causal "
        "32-bucket learned relative-distance bias"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (2400,)
    base.learning_rate = matched_learning_rate
    base.make_model = candidate_make_model
    base.checkpoint_payload = candidate_checkpoint_payload
    base.load_data = load_train_validation
    base.SOURCE_FILES = base.SOURCE_FILES + (
        CONFIG.as_posix(), "student_stage216_relative_bias.py",
        "scripts/preflight_stage216_relative_bias.py",
        "scripts/run_stage216_relative_bias_pilot.py",
        "scripts/run_stage216_pilot_windows.ps1",
        "scripts/start_stage216_pilot_windows.ps1",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "docs/STAGE216_RELATIVE_BIAS_PLAN_20260928.md",
        "tests/test_stage216_relative_bias.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
