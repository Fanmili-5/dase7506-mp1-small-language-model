"""Fixed 2,400-step Stage54 pilot with training-only block drop path 0.10."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model as original_make_model, sha
from drop_path_pilot import attach_drop_path
from scripts import train_stage54_hybrid_conv_rdrop as base
from train_experiment import learning_rate as stage54_learning_rate

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage184_drop_path_hybrid_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
STAGE54_TOTAL_STEPS = 7200
DROP_PATH_RATE = 0.10


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return stage54_learning_rate(step, STAGE54_TOTAL_STEPS, *args)


def make_drop_path_model(implementation, config, device):
    model, implementation_sha = original_make_model(implementation, config, device)
    attach_drop_path(model, DROP_PATH_RATE)
    return model, implementation_sha


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    control = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    if config.get("drop_path_rate") != DROP_PATH_RATE:
        raise ValueError("Unexpected drop-path rate")
    if {key: value for key, value in config.items() if key != "drop_path_rate"} != control:
        raise ValueError("Stage184 config must differ from Stage54 only in drop_path_rate")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 matched R-Drop control at 2,400 updates; only "
        "training-time block drop path 0.10 is added"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.learning_rate = matched_learning_rate
    base.make_model = make_drop_path_model
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES if name != CONTROL.as_posix()
    ) + (CONFIG.as_posix(), "drop_path_pilot.py", "scripts/run_stage184_drop_path_pilot.py")
    base.main()


if __name__ == "__main__":
    main()
