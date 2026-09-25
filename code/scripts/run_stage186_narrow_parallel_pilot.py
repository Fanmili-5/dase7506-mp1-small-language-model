"""Matched 2,400-step Stage54 pilot after the fixed Stage186 resource pass."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model as original_make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from train_experiment import checkpoint_payload as original_checkpoint_payload
from train_experiment import learning_rate as stage54_learning_rate

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage186_narrow_parallel_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
PREFLIGHT = ROOT / "runs/stage186-narrow-preflight/result.json"
IMPLEMENTATION = "student_stage186_narrow_parallel_rdrop"
STAGE54_TOTAL_STEPS = 7200


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return stage54_learning_rate(step, STAGE54_TOTAL_STEPS, *args)


def make_narrow_model(_implementation, config, device):
    return original_make_model(IMPLEMENTATION, config, device)


def stage186_checkpoint_payload(model, _implementation, config, seed, train_tokens):
    return original_checkpoint_payload(model, IMPLEMENTATION, config, seed, train_tokens)


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 training implementation changed")
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    control = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    if (config.get("parallel_heads") != 2
            or config.get("parallel_head_dim") != 36
            or {key: value for key, value in config.items()
                if key not in {"parallel_heads", "parallel_head_dim"}} != control):
        raise ValueError("Stage186 must differ only by its fixed narrow attention branch")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if (preflight.get("eligible_for_pilot") is not True
            or preflight.get("test_scored") is not False
            or preflight.get("config_sha256") != sha(ROOT / CONFIG)
            or preflight.get("parallel_module_sha256") != sha(
                ROOT / "student_stage186_narrow_parallel.py")):
        raise ValueError("Input-only resource screen does not authorize training")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 matched R-Drop control at 2,400 updates; only "
        "four quarter-width global branches are added beside local blocks"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.learning_rate = matched_learning_rate
    base.make_model = make_narrow_model
    base.checkpoint_payload = stage186_checkpoint_payload
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES if name != CONTROL.as_posix()
    ) + (
        CONFIG.as_posix(), "student_stage186_narrow_parallel.py",
        "student_stage186_narrow_parallel_rdrop.py",
        "student_stage186_narrow_parallel_structured.py",
        "scripts/run_stage186_narrow_parallel_pilot.py",
    )
    base.main()


if __name__ == "__main__":
    main()
