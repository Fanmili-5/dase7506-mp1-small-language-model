"""Inspect Stage105 FFN singular spectra before attempting FP32 low-rank inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import make_model, setup, sha


STAGE105_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"
RANKS = (128, 160, 192, 224, 256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.checkpoint) != STAGE105_SHA or args.output.exists():
        raise ValueError("Unexpected source checkpoint or existing output")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    rows = []
    for index, block in enumerate(model.neural.blocks):
        for name in ("input", "output"):
            matrix = getattr(block.mlp, name).weight.detach().double()
            singular = torch.linalg.svdvals(matrix)
            energy = singular.square().cumsum(0) / singular.square().sum()
            rows.append({
                "block": index + 1,
                "matrix": name,
                "shape": list(matrix.shape),
                "relative_frobenius_error": {
                    str(rank): float((1 - energy[min(rank, len(energy)) - 1]).clamp_min(0).sqrt())
                    for rank in RANKS
                },
            })
    result = {
        "purpose": "training_weight_spectrum_only_no_validation_or_test_scoring",
        "source_checkpoint_sha256": STAGE105_SHA,
        "ranks": RANKS,
        "matrices": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
