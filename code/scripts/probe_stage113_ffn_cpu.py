"""Estimate the exact Stage105 CPU forward saving from removing one FFN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import load_data, make_model, setup, sha, windows
from scripts.screen_stage112_block_ablation import ZeroFFN


CHECKPOINT_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"


def timed(model, ids, repetitions=8):
    with torch.inference_mode():
        for _ in range(2):
            model.predict_log_probs(ids)
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            model.predict_log_probs(ids)
            samples.append(time.perf_counter() - started)
    return sorted(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Unexpected Stage105 checkpoint")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    ids, _ = next(windows(load_data()["validation"][0], 32))
    original = timed(model, ids)
    full_ffn = model.neural.blocks[-1].mlp
    model.neural.blocks[-1].mlp = ZeroFFN()
    pruned = timed(model, ids)
    model.neural.blocks[-1].mlp = full_ffn
    full_block = model.neural.blocks[4]
    model.neural.blocks[4] = torch.nn.Identity()
    block_pruned = timed(model, ids)
    model.neural.blocks[4] = full_block
    median = lambda values: (values[3] + values[4]) / 2
    print(json.dumps(dict(original_seconds=original, pruned_seconds=pruned,
                          block_pruned_seconds=block_pruned,
                          original_median=median(original),
                          pruned_median=median(pruned),
                          relative_forward_time=median(pruned) / median(original),
                          block_pruned_median=median(block_pruned),
                          block_relative_forward_time=median(block_pruned) / median(original),
                          source_sha256=sha(Path(__file__)),
                          no_test_scoring=True), indent=2))


if __name__ == "__main__":
    main()
