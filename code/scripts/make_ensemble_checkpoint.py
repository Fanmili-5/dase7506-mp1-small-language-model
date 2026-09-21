"""Package two existing student checkpoints as one auditable predictor."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import PROTOCOL, make_model, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--primary-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.primary_weight < 1.0:
        parser.error("--primary-weight must lie strictly between zero and one")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    sources = [
        torch.load(args.primary, map_location="cpu", weights_only=True),
        torch.load(args.auxiliary, map_location="cpu", weights_only=True),
    ]
    if any(source["protocol"] != PROTOCOL for source in sources):
        raise ValueError("A checkpoint belongs to a different course protocol.")
    if any(source["implementation"] != "student" for source in sources):
        raise ValueError("This ensemble packager accepts student.py checkpoints only.")

    config = {
        "vocab": 2048,
        "context": 256,
        "member_configs": [source["config"] for source in sources],
        "weights": [args.primary_weight, 1.0 - args.primary_weight],
    }
    ensemble, _ = make_model("student_ensemble", config, torch.device("cpu"))
    for member, source in zip(ensemble.members, sources, strict=True):
        member.load_state_dict(source["model"])
    train_targets = [int(source.get("train_tokens", 0)) for source in sources]
    payload = {
        "protocol": PROTOCOL,
        "implementation": "student_ensemble",
        "config": config,
        "model": ensemble.state_dict(),
        "seed": [source.get("seed") for source in sources],
        "train_tokens": sum(train_targets),
        "ensemble_ancestry": {
            "source_checkpoint_sha256": [sha(args.primary), sha(args.auxiliary)],
            "source_train_targets": train_targets,
            "weights": config["weights"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes, sha256={sha(args.output)})")


if __name__ == "__main__":
    main()
