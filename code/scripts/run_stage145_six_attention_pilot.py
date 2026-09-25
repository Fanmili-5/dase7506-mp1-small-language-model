"""Matched 2,400-step R-Drop pilot; changes only Stage54 token mixing."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import sha
from scripts import train_stage54_hybrid_conv_rdrop as base


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    base.CONFIG = Path("configs/stage145_six_attention_rdrop.json")
    base.COMPARISON = (
        "Stage54 seed-17 matched R-Drop control at 2,400 updates; only "
        "conv_layers changes [2,4,6,8] to [4,8]"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != "configs/stage54_hybrid_conv_rdrop.json"
    ) + (base.CONFIG.as_posix(), "scripts/run_stage145_six_attention_pilot.py")
    base.main()


if __name__ == "__main__":
    main()
