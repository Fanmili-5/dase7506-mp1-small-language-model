"""Run the fixed Stage54 R-Drop protocol on the Stage58 local-heavy graph."""
from pathlib import Path

import train_stage54_hybrid_conv_rdrop as train

train.CONFIG = Path("configs/stage58_local_heavy_rdrop.json")
train.COMPARISON = (
    "Stage54 matched R-Drop recipe; width320 and six conv/two attention layers"
)
train.SOURCE_FILES = (
    "student_hybrid_conv_rdrop.py", "student_hybrid_conv_structured.py",
    "student_rdrop_multi_token.py", "student_multi_token.py",
    "student_deep_supervision.py", "student_regularized.py",
    "student_structured.py", "student.py",
    "scripts/train_stage59_local_heavy_rdrop.py",
    "scripts/train_stage54_hybrid_conv_rdrop.py", "train_experiment.py",
    "evaluate.py", "common.py", train.CONFIG.as_posix(),
    "data/manifest.json", "data/tokenizer.json",
)


if __name__ == "__main__":
    train.main()
