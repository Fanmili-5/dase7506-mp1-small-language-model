"""Preserve teacher ancestry when averaging checkpoints from one KD trajectory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from common import sha
from scripts.average_checkpoints import average_checkpoints


def average_with_provenance(paths):
    result = average_checkpoints(paths)
    provenance = [torch.load(p, map_location="cpu", weights_only=True)["distillation_provenance"] for p in paths]
    if any(value != provenance[0] for value in provenance[1:]):
        raise ValueError("Cannot average across different distillation recipes or teachers")
    result["distillation_provenance"] = provenance[0]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = average_with_provenance(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"Wrote {args.output}, sha256={sha(args.output)}")


if __name__ == "__main__":
    main()
