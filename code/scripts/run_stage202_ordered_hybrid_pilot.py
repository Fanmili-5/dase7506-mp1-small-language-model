"""Matched Stage54 seed-17 pilot with local layers before global attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import learning_rate as stage54_learning_rate

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage202_bottom_local_top_global.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return stage54_learning_rate(step, 7200, *args)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 trainer changed")
    candidate = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    control = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    changed = {key for key in candidate.keys() | control.keys()
               if candidate.get(key) != control.get(key)}
    if changed != {"conv_layers"} or candidate["conv_layers"] != [1, 2, 3, 4]:
        raise ValueError("Stage202 must change only the fixed block order")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (preflight.get("training_gate_passed") is not True
            or preflight.get("no_data_split_opened") is not True
            or preflight.get("equal_parameter_count") is not True
            or preflight.get("config_sha256") != sha(ROOT / CONFIG)
            or preflight.get("source_sha256") != sha(
                ROOT / "scripts/preflight_stage202_ordered_hybrid.py")):
        raise ValueError("Stage202 synthetic preflight did not pass")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 matched seed-17 first 2400 steps; only four local/global "
        "block positions change, from alternating to local-first/global-later"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (2400,)
    base.learning_rate = matched_learning_rate
    base.load_data = load_train_validation
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES if name != CONTROL.as_posix()
    ) + (
        CONFIG.as_posix(), "scripts/preflight_stage202_ordered_hybrid.py",
        "scripts/run_stage202_ordered_hybrid_pilot.py",
        "scripts/start_stage202_ordered_hybrid_windows.ps1",
        "scripts/run_stage202_ordered_hybrid_windows.ps1",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "docs/STAGE202_BOTTOM_LOCAL_TOP_GLOBAL_PLAN_20260928.md",
        "tests/test_stage202_ordered_hybrid.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
