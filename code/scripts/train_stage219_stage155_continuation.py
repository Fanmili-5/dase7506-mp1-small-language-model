"""Fixed 4,800-step train/validation-only continuation of Stage155 average."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import sha
from scripts import train_stage56_hybrid_conv_continuation as base
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation

BASE_SHA = "85a04b88eaca1899a1656771bd12a1a5e5d29414b9c4916afd8ebd2bdf462c1d"
START_SHA = "c2fe32ba15b0f13b11b43247f058971dac5717ac37fd2acda1bbaa173277bce1"
CONFIG = Path("configs/stage155_neural_budget_rdrop.json")


def main() -> None:
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Stage56 continuation recipe changed")
    base.START_SHA = START_SHA
    base.START_STAGE = "Stage155 seed-17 last-five training average"
    base.COMPARISON = (
        "Stage155 4,800-step low-LR continuation to test the training-budget "
        "confound; compare full validation with Stage155 and Stage143"
    )
    base.CONFIG = CONFIG
    base.load_data = load_train_validation
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != "configs/stage54_hybrid_conv_rdrop.json"
    ) + (
        CONFIG.as_posix(), "scripts/train_stage193_fresh_mixture_pilot.py",
        "scripts/train_stage219_stage155_continuation.py",
        "docs/STAGE219_EQUAL_BUDGET_CONTINUATION_PLAN_20260928.md",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    base.main()


if __name__ == "__main__":
    main()
