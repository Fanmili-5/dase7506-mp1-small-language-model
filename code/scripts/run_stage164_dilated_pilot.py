"""Matched 2,400-step Stage54 pilot with multiscale causal convolutions."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model as original_make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from train_experiment import checkpoint_payload as original_checkpoint_payload
from train_experiment import learning_rate as stage54_learning_rate

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
STAGE54_TOTAL_STEPS = 7200
IMPLEMENTATION = "student_dilated_hybrid_rdrop"


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return stage54_learning_rate(step, STAGE54_TOTAL_STEPS, *args)


def matched_model(requested: str, config: dict, device):
    if requested != "student_hybrid_conv_rdrop":
        raise ValueError("Unexpected Stage54 implementation request")
    return original_make_model(IMPLEMENTATION, config, device)


def matched_checkpoint(model, requested: str, config: dict, seed: int, train_tokens: int):
    if requested != "student_hybrid_conv_rdrop":
        raise ValueError("Unexpected Stage54 checkpoint implementation")
    return original_checkpoint_payload(model, IMPLEMENTATION, config, seed, train_tokens)


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    base.CONFIG = Path("configs/stage164_dilated_hybrid_rdrop.json")
    base.COMPARISON = (
        "Stage54 seed-17 matched 2,400-update R-Drop control; only "
        "conv dilations [1,1,1,1] become [1,2,4,8]"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.learning_rate = matched_learning_rate
    base.make_model = matched_model
    base.checkpoint_payload = matched_checkpoint
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != "configs/stage54_hybrid_conv_rdrop.json"
    ) + (
        base.CONFIG.as_posix(),
        "student_dilated_hybrid_structured.py",
        "student_dilated_hybrid_rdrop.py",
        "scripts/run_stage164_dilated_pilot.py",
    )
    base.main()


if __name__ == "__main__":
    main()
