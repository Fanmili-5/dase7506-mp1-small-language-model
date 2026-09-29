"""Fresh 7,200-step seed-17 Stage155 R-Drop trajectory after pilot gate."""
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
    base.CONFIG = Path("configs/stage155_neural_budget_rdrop.json")
    base.COMPARISON = (
        "Fresh Stage155 7,200-step seed-17 full schedule after a fixed "
        "0.022450508 BPB matched 2,400-step pilot improvement"
    )
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES
        if name != "configs/stage54_hybrid_conv_rdrop.json"
    ) + (base.CONFIG.as_posix(), "scripts/run_stage155_neural_budget_full.py")
    base.main()


if __name__ == "__main__":
    main()
