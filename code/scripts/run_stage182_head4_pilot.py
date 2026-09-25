"""Matched 2,400-step R-Drop pilot; change only Stage54 heads 8 to 4."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from train_experiment import learning_rate as stage54_learning_rate

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage182_head4_hybrid_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
STAGE54_TOTAL_STEPS = 7200


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return stage54_learning_rate(step, STAGE54_TOTAL_STEPS, *args)


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    control = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    if config.get("heads") != 4 or control.get("heads") != 8:
        raise ValueError("Unexpected head counts")
    if {key: value for key, value in config.items() if key != "heads"} != {
            key: value for key, value in control.items() if key != "heads"}:
        raise ValueError("Stage182 must differ from Stage54 only in head count")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 matched R-Drop control at 2,400 updates; only "
        "attention heads change from 8 to 4"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.learning_rate = matched_learning_rate
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != CONTROL.as_posix()
    ) + (CONFIG.as_posix(), "scripts/run_stage182_head4_pilot.py")
    base.main()


if __name__ == "__main__":
    main()
