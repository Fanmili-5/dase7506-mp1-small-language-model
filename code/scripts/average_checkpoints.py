"""Average checkpoints from one training trajectory into one predictor."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import PROTOCOL, sha


def average_checkpoints(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("At least two checkpoints are required for averaging.")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    first = checkpoints[0]
    for checkpoint in checkpoints:
        if checkpoint["protocol"] != PROTOCOL:
            raise ValueError("A checkpoint belongs to a different course protocol.")
        if checkpoint["implementation"] != first["implementation"] or checkpoint["config"] != first["config"]:
            raise ValueError("Checkpoint implementations and configs must match exactly.")
        if checkpoint.get("seed") != first.get("seed"):
            raise ValueError("Only checkpoints from the same seeded trajectory may be averaged.")
        if checkpoint["model"].keys() != first["model"].keys():
            raise ValueError("Checkpoint state dictionaries do not have identical keys.")

    averaged = {}
    count = len(checkpoints)
    for key, reference in first["model"].items():
        tensors = [checkpoint["model"][key] for checkpoint in checkpoints]
        if any(tensor.shape != reference.shape or tensor.dtype != reference.dtype for tensor in tensors):
            raise ValueError(f"State tensor metadata differs for {key}.")
        if reference.is_floating_point():
            total = torch.zeros_like(reference, dtype=torch.float64)
            for tensor in tensors:
                total.add_(tensor.double())
            averaged[key] = (total / count).to(reference.dtype)
        else:
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating state differs for {key}.")
            averaged[key] = reference.clone()

    train_targets = [int(checkpoint.get("train_tokens", 0)) for checkpoint in checkpoints]
    return {
        "protocol": PROTOCOL,
        "implementation": first["implementation"],
        "config": first["config"],
        "model": averaged,
        "seed": first.get("seed"),
        "train_tokens": max(train_targets),
        "averaging_ancestry": {
            "method": "uniform_same_trajectory_parameter_average",
            "source_checkpoint_sha256": [sha(path) for path in paths],
            "source_train_targets": train_targets,
            "source_count": count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    payload = average_checkpoints(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes, sha256={sha(args.output)})")


if __name__ == "__main__":
    main()
