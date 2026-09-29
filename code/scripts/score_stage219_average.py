"""Score the fixed Stage219 averaged checkpoint on validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, device_metrics, make_model, setup, sha
from evaluate import score
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation
from train_experiment import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = json.loads((ROOT / "configs/stage155_neural_budget_rdrop.json").read_text())
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_rdrop"
            or payload.get("config") != config
            or payload.get("seed") != 17
            or payload.get("train_tokens") != 98_304_000
            or payload.get("averaging_ancestry", {}).get("source_count") != 5):
        raise ValueError("Expected the prespecified Stage219 average")
    device, precision = setup("cuda", "fp32", 4)
    model, implementation_sha = make_model(payload["implementation"], config, device)
    model.load_state_dict(payload["model"], strict=True)
    result = score(model, *load_train_validation()["validation"], device, precision)
    result.pop("window_nll_nats")
    result.update(
        protocol=PROTOCOL, split="validation", precision=precision,
        purpose="stage219_prespecified_last_five_validation_only",
        no_test_scoring=True, checkpoint_sha256=sha(args.checkpoint),
        implementation_sha256=implementation_sha,
        evaluator_sha256=sha(ROOT / "evaluate.py"),
        tokenizer_sha256=sha(ROOT / "data/tokenizer.json"),
        **device_metrics(device),
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
