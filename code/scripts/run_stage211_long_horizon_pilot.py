"""Fixed same-target Stage211 long-horizon auxiliary supervision pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.preflight_stage211_long_horizon import build_matched_candidate
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import learning_rate as original_learning_rate


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage211_long_horizon_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return original_learning_rate(step, 7200, *args)


def candidate_make_model(implementation: str, config: dict, device):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed implementation")
    control_config = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    control, candidate, digest = build_matched_candidate(
        config, control_config, device)
    del control
    return candidate, digest


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
    if (changed != {"future_prediction_offsets"}
            or config["future_prediction_offsets"] != [2, 3, 8, 16, 32, 64]):
        raise ValueError("Stage211 changed outside its fixed auxiliary offsets")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (preflight.get("admit_matched_training_pilot") is not True
            or preflight.get("no_data_split_opened") is not True
            or preflight.get("config_sha256") != sha(ROOT / CONFIG)
            or preflight.get("source_sha256") != sha(
                ROOT / "scripts/preflight_stage211_long_horizon.py")):
        raise ValueError("Fixed Stage211 synthetic preflight did not pass")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 first 2400 steps, batch32, exact 7200-step LR prefix; "
        "future labels extend from [2,3] to [2,3,8,16,32,64] with the same "
        "total 0.2 training-only auxiliary weight"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (2400,)
    base.learning_rate = matched_learning_rate
    base.make_model = candidate_make_model
    base.load_data = load_train_validation
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES if name != CONTROL.as_posix()
    ) + (
        CONFIG.as_posix(),
        "scripts/preflight_stage211_long_horizon.py",
        "scripts/run_stage211_long_horizon_pilot.py",
        "scripts/run_stage211_pilot_windows.ps1",
        "scripts/start_stage211_pilot_windows.ps1",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "docs/STAGE211_LONG_HORIZON_AUXILIARY_PLAN_20260928.md",
        "tests/test_stage211_long_horizon.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
