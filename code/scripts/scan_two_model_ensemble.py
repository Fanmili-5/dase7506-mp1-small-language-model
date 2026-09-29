"""Scan two-model probability-mixture weights on the validation split only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import PROTOCOL, ROOT, autocast, load_data, make_model, setup, sha, windows


def parse_weights(text: str) -> list[float]:
    weights = sorted({float(value) for value in text.split(",")})
    if not weights or any(value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("Every primary weight must lie in [0, 1].")
    return weights


@torch.no_grad()
def target_log_probs(checkpoint: dict, tokens: torch.Tensor, device: torch.device,
                     precision: str, batch_size: int) -> torch.Tensor:
    model, _ = make_model(checkpoint["implementation"], checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    for x, y in windows(tokens, batch_size):
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            logp = model.predict_log_probs(x).float()
        selected = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        rows.append(selected[y != -100].double().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(rows)


def mixture_log_probs(primary: torch.Tensor, auxiliary: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0.0:
        return auxiliary
    if weight == 1.0:
        return primary
    return torch.logaddexp(primary + math.log(weight), auxiliary + math.log1p(-weight))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--primary-weights", default="0.70,0.75,0.80,0.85,0.90,0.95,1.0")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "auto"], default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")

    device, precision = setup(args.device, args.precision, args.threads)
    checkpoints = [
        torch.load(args.primary, map_location="cpu", weights_only=True),
        torch.load(args.auxiliary, map_location="cpu", weights_only=True),
    ]
    if any(checkpoint["protocol"] != PROTOCOL for checkpoint in checkpoints):
        raise ValueError("A checkpoint belongs to a different course protocol.")
    data = load_data()
    tokens, byte_count = data["validation"]
    primary = target_log_probs(checkpoints[0], tokens, device, precision, args.batch_size)
    auxiliary = target_log_probs(checkpoints[1], tokens, device, precision, args.batch_size)
    if len(primary) != len(auxiliary) or len(primary) != len(tokens) - 1:
        raise ValueError("Member target accounting differs from the fixed validation protocol.")

    rows = []
    for weight in parse_weights(args.primary_weights):
        mixed = mixture_log_probs(primary, auxiliary, weight)
        nll = -mixed.sum().item()
        rows.append({
            "primary_weight": weight,
            "auxiliary_weight": 1.0 - weight,
            "bpb": nll / math.log(2.0) / byte_count,
            "token_ppl": math.exp(nll / len(mixed)),
            "nll_nats": nll,
        })
    best = min(rows, key=lambda row: row["bpb"])
    result = {
        "protocol": PROTOCOL,
        "split": "validation",
        "selection_warning": "Mixture weights are validation-selected; do not use test results for this choice.",
        "device": str(device),
        "precision": precision,
        "targets": len(primary),
        "utf8_bytes": byte_count,
        "primary": {
            "path": str(args.primary),
            "sha256": sha(args.primary),
            "implementation": checkpoints[0]["implementation"],
        },
        "auxiliary": {
            "path": str(args.auxiliary),
            "sha256": sha(args.auxiliary),
            "implementation": checkpoints[1]["implementation"],
        },
        "results": rows,
        "best": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
