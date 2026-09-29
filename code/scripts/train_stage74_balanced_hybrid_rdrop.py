"""Run the Stage54 matched R-Drop trainer with the admitted Stage72 graph."""
from pathlib import Path

import train_stage54_hybrid_conv_rdrop as trainer


CONFIG = Path("configs/stage72_balanced_hybrid_rdrop.json")
WRAPPER = "scripts/train_stage74_balanced_hybrid_rdrop.py"

trainer.CONFIG = CONFIG
trainer.COMPARISON = (
    "Stage54 matched 7200-update R-Drop recipe; only the admitted Stage72 "
    "global/local backbone allocation changes"
)
trainer.SOURCE_FILES = tuple(
    name for name in trainer.SOURCE_FILES
    if name != "configs/stage54_hybrid_conv_rdrop.json"
) + (CONFIG.as_posix(), WRAPPER)


if __name__ == "__main__":
    trainer.main()
