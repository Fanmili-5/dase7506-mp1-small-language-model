"""Fixed Stage54-matched 2,400-step train-derived bigram input pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import make_model as original_make_model, sha
from scripts import train_stage54_hybrid_conv_rdrop as base
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from student_stage223_bigram_input import build_train_bigram_rank_map
from train_experiment import (checkpoint_payload as original_checkpoint_payload,
                              learning_rate as original_learning_rate)


BASE_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage223_bigram_input_rdrop.json")
CONTROL = Path("configs/stage54_hybrid_conv_rdrop.json")
IMPLEMENTATION = "student_stage223_bigram_rdrop"
EXPECTED_MAP_SHA = "7ea76a7bdd674bb5a2149df4bc23b9d658b170fe06d64b0a2efad9680d23b979"
EXPECTED_GRAPH_SHA = "089b0f049cebd01f59c52b45d7ce1048a210b833918275ee2a881f5a35a8e5bb"


def matched_learning_rate(step: int, _pilot_steps: int, *args):
    return original_learning_rate(step, 7200, *args)


def candidate_make_model(implementation: str, config: dict, device):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed implementation")
    model, digest = original_make_model(IMPLEMENTATION, config, device)
    mapping = build_train_bigram_rank_map(load_train_validation()["train"][0])
    map_sha = hashlib.sha256(mapping.numpy().tobytes()).hexdigest()
    if map_sha != EXPECTED_MAP_SHA:
        raise ValueError("Train-derived bigram map changed")
    model.set_bigram_rank_map(mapping)
    return model, digest


def candidate_checkpoint_payload(model, implementation: str, *args):
    if implementation != "student_hybrid_conv_rdrop":
        raise ValueError("Matched trainer changed checkpoint implementation")
    return original_checkpoint_payload(model, IMPLEMENTATION, *args)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sha(Path(base.__file__)) != BASE_SHA:
        raise ValueError("Matched Stage54 trainer changed")
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    control = json.loads((ROOT / CONTROL).read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control.keys()
               if config.get(key) != control.get(key)}
    if changed != {"bigram_top_k", "bigram_dim"}:
        raise ValueError("Stage223 config changed outside fixed bigram input")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8-sig"))
    expected = {
        "config_sha256": sha(ROOT / CONFIG),
        "source_sha256": sha(ROOT / "scripts/preflight_stage223_bigram_input.py"),
        "core_sha256": sha(ROOT / "student_stage223_bigram_input.py"),
        "training_model_sha256": sha(ROOT / "student_stage223_bigram_rdrop.py"),
        "inference_model_sha256": sha(ROOT / "student_stage223_bigram_structured.py"),
        "train_bigram_rank_map_sha256": EXPECTED_MAP_SHA,
        "candidate_graph_sha256": EXPECTED_GRAPH_SHA,
    }
    if (preflight.get("admit_matched_training_pilot") is not True
            or preflight.get("no_validation_or_test_scoring") is not True
            or any(preflight.get(key) != value for key, value in expected.items())):
        raise ValueError("Fixed Stage223 preflight did not pass or source changed")
    base.CONFIG = CONFIG
    base.COMPARISON = (
        "Stage54 seed-17 first 2400 steps, batch32, exact 7200-step LR prefix; "
        "only fixed train-derived top-16384 bigram input embeddings differ"
    )
    base.STEPS = 2400
    base.TARGETS = base.STEPS * base.BATCH * 256
    base.AVERAGE_STEPS = (2400,)
    base.learning_rate = matched_learning_rate
    base.make_model = candidate_make_model
    base.checkpoint_payload = candidate_checkpoint_payload
    base.load_data = load_train_validation
    base.SOURCE_FILES = base.SOURCE_FILES + (
        CONFIG.as_posix(), CONTROL.as_posix(),
        "student_stage223_bigram_input.py",
        "student_stage223_bigram_rdrop.py",
        "student_stage223_bigram_structured.py",
        "scripts/preflight_stage223_bigram_input.py",
        "scripts/run_stage223_bigram_pilot.py",
        "scripts/run_stage223_pilot_windows.ps1",
        "scripts/start_stage223_pilot_windows.ps1",
        "scripts/train_stage193_fresh_mixture_pilot.py",
        "docs/STAGE223_TRAIN_BIGRAM_INPUT_PLAN_20260928.md",
        "tests/test_stage223_bigram_input.py",
        "results/stage223-bigram-preflight.json",
        "data/wikitext_train.txt", "data/wikitext_validation.txt",
    )
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
