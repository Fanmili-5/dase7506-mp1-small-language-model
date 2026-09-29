"""Matched 2,400-step seed-17 pilot for the seven-block shared-trunk LM."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from train_experiment import checkpoint_payload, learning_rate as stage54_learning_rate


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
STAGE54_TOTAL_STEPS = 7200
IMPLEMENTATION = "student_stage154_shared_last_block"


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return stage54_learning_rate(step, STAGE54_TOTAL_STEPS, *args)


def make_stage154_model(_implementation, config, device):
    return make_model(IMPLEMENTATION, config, device)


def stage154_checkpoint_payload(model, _implementation, config, seed, tokens):
    return checkpoint_payload(model, IMPLEMENTATION, config, seed, tokens)


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    base.CONFIG = Path("configs/stage154_shared_last_block_rdrop.json")
    base.COMPARISON = (
        "Stage54 seed-17 matched R-Drop control at 2,400 updates; "
        "seven shared blocks plus heterogeneous eighth-block mixture"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.learning_rate = matched_learning_rate
    base.make_model = make_stage154_model
    base.checkpoint_payload = stage154_checkpoint_payload
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != "configs/stage54_hybrid_conv_rdrop.json"
    ) + (
        base.CONFIG.as_posix(),
        "student_stage154_shared_last_block.py",
        "scripts/run_stage154_shared_last_block_pilot.py",
    )
    base.main()


if __name__ == "__main__":
    main()
