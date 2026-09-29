"""Matched 2,400-step Stage177 quality pilot after its new GPU preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model as original_make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import checkpoint_payload as original_checkpoint_payload
from train_experiment import learning_rate as stage54_learning_rate

BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage177_parallel_mixer_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
IMPLEMENTATION = "student_stage177_parallel_mixer_rdrop"
STAGE54_TOTAL_STEPS = 7200


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return stage54_learning_rate(step, STAGE54_TOTAL_STEPS, *args)


def stage199_model(_implementation, config, device):
    return original_make_model(IMPLEMENTATION, config, device)


def stage199_checkpoint_payload(model, _implementation, config, seed, train_tokens):
    return original_checkpoint_payload(model, IMPLEMENTATION, config, seed, train_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 trainer changed")
    if json.loads((ROOT / CONFIG).read_text(encoding="utf-8")) != json.loads(
            (ROOT / CONTROL).read_text(encoding="utf-8")):
        raise ValueError("Stage177 pilot must change only model implementation")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    if (preflight.get("implementation") != IMPLEMENTATION
            or preflight.get("training_gate_passed") is not True
            or preflight.get("no_data_split_opened") is not True
            or preflight.get("config_sha256") != sha(ROOT / CONFIG)
            or preflight.get("implementation_sha256") != sha(
                ROOT / "student_stage177_parallel_mixer_rdrop.py")):
        raise ValueError("Stage199 fixed-batch GPU preflight did not pass")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 matched 2,400-step R-Drop control; only four "
        "full-width parallel attention paths added beside conv blocks"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
    base.learning_rate = matched_learning_rate
    base.make_model = stage199_model
    base.checkpoint_payload = stage199_checkpoint_payload
    base.load_data = load_train_validation
    base.SOURCE_FILES = tuple(
        name for name in base.SOURCE_FILES if name != CONTROL.as_posix()
    ) + (
        CONFIG.as_posix(), "student_stage177_parallel_mixer.py",
        "student_stage177_parallel_mixer_rdrop.py",
        "student_stage177_parallel_mixer_structured.py",
        "scripts/preflight_stage199_parallel_gpu.py",
        "scripts/run_stage199_parallel_pilot.py",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
