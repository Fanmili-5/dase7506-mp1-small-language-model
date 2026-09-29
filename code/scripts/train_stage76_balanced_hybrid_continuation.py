"""Run the matched Stage56 continuation on the Stage74 training average."""
from pathlib import Path

import train_stage56_hybrid_conv_continuation as trainer


CONFIG = Path("configs/stage72_balanced_hybrid_rdrop.json")
WRAPPER = "scripts/train_stage76_balanced_hybrid_continuation.py"

trainer.START_SHA = "a6c7d296ec68e971691d6d3beac8b8a3ab9292676845feca1031346fc4c250ec"
trainer.START_STAGE = "Stage74 fixed five-checkpoint training average"
trainer.DROPOUT_RNG_SEED = 74017
trainer.COMPARISON = (
    "matched low-LR continuation of the admitted Stage74 balanced hybrid average"
)
trainer.CONFIG = CONFIG
trainer.SOURCE_FILES = tuple(
    name for name in trainer.SOURCE_FILES
    if name != "configs/stage54_hybrid_conv_rdrop.json"
) + (CONFIG.as_posix(), WRAPPER)


if __name__ == "__main__":
    trainer.main()
