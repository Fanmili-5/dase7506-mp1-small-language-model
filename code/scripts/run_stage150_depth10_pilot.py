"""Matched 2,400-step R-Drop pilot; add two alternating Stage54 blocks."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from train_experiment import learning_rate as stage54_learning_rate


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
STAGE54_TOTAL_STEPS = 7200


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    """Keep Stage54's 7,200-step LR curve through pilot step 2,400."""
    return stage54_learning_rate(step, STAGE54_TOTAL_STEPS, *args)


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    base.CONFIG = Path("configs/stage150_depth10_hybrid_rdrop.json")
    base.COMPARISON = (
        "Stage54 seed-17 R-Drop control at 2,400 updates; only depth eight "
        "to ten and an alternating attention/conv pair added"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.learning_rate = matched_learning_rate
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != "configs/stage54_hybrid_conv_rdrop.json"
    ) + (base.CONFIG.as_posix(), "scripts/run_stage150_depth10_pilot.py")
    base.main()


if __name__ == "__main__":
    main()
