"""Fixed Stage54 training trajectory with only main dropout changed to 0.20."""
from pathlib import Path

import train_stage54_hybrid_conv_rdrop as base


base.CONFIG = Path("configs/stage179_hybrid_dropout20_rdrop.json")
base.COMPARISON = (
    "Stage54 seed-17 7200-step same-target control; only main dropout 0.10 to 0.20"
)
base.SOURCE_FILES = tuple(
    base.CONFIG.as_posix() if name == "configs/stage54_hybrid_conv_rdrop.json" else name
    for name in base.SOURCE_FILES
) + ("scripts/train_stage179_dropout20.py",)


if __name__ == "__main__":
    base.main()
