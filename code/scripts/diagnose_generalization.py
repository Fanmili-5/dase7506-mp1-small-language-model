"""FP32 train/validation error diagnostics; never scores test or changes weights."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from common import PROTOCOL, load_data, make_model, setup, sha, windows


def sampled_train_windows(tokens, count, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(len(tokens) - 256, (count,), generator=generator)
    for offset in range(0, count, batch_size):
        ids = tokens[starts[offset:offset + batch_size, None] + torch.arange(257)]
        yield ids[:, :-1], ids[:, 1:]


@torch.inference_mode()
def diagnose(model, batches, frequencies, device):
    totals = {}
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logp = model.predict_log_probs(x).float()
        valid = y != -100
        losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).double()
        if not torch.isfinite(losses[valid]).all():
            raise ValueError("Non-finite losses.")
        positions = torch.arange(x.shape[1], device=device)[None, :]
        prefix = torch.ones(x.shape[1], x.shape[1], device=device, dtype=torch.bool).tril()
        seen = ((y.unsqueeze(-1) == x.unsqueeze(1)) & prefix).any(-1)
        counts = frequencies[y.clamp_min(0)]
        groups = {
            "all": valid,
            "positions_1_16": valid & (positions < 16),
            "positions_17_64": valid & (positions >= 16) & (positions < 64),
            "positions_65_256": valid & (positions >= 64),
            "target_seen_in_prefix": valid & seen,
            "target_not_seen_in_prefix": valid & ~seen,
            "train_frequency_lt_100": valid & (counts < 100),
            "train_frequency_100_999": valid & (counts >= 100) & (counts < 1000),
            "train_frequency_1000_9999": valid & (counts >= 1000) & (counts < 10000),
            "train_frequency_ge_10000": valid & (counts >= 10000),
        }
        for name, mask in groups.items():
            row = totals.setdefault(name, {"targets": 0, "nll_nats": 0.0})
            row["targets"] += int(mask.sum())
            row["nll_nats"] += float(losses[mask].sum())
    for row in totals.values():
        row["mean_nll_nats"] = row["nll_nats"] / row["targets"] if row["targets"] else None
        row["token_ppl"] = math.exp(row["mean_nll_nats"]) if row["targets"] else None
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-windows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.output.exists() or args.train_windows <= 0:
        parser.error("Output must be new and train-windows must be positive.")
    device, _ = setup(args.device, "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != PROTOCOL:
        raise ValueError("Unexpected protocol.")
    model, implementation_sha = make_model(checkpoint["implementation"], checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = load_data()
    frequencies = torch.bincount(data["train"][0], minlength=2048).to(device)
    training = diagnose(model, sampled_train_windows(data["train"][0], args.train_windows, 32, args.seed), frequencies, device)
    print(json.dumps({"sampled_train": training["all"]}), flush=True)
    validation = diagnose(model, windows(data["validation"][0], 32), frequencies, device)
    if validation["all"]["targets"] != len(data["validation"][0]) - 1:
        raise ValueError("Incomplete validation coverage.")
    result = {
        "protocol": PROTOCOL, "precision": "fp32", "device": str(device),
        "checkpoint_sha256": sha(args.checkpoint), "implementation_sha256": implementation_sha,
        "diagnostic_sha256": sha(Path(__file__)), "sampling_seed": args.seed,
        "training_sample_windows": args.train_windows,
        "training_sample_targets": args.train_windows * 256,
        "training_split_tokens": len(data["train"][0]),
        "note": "Train uses uniformly sampled independent windows and is an in-sample diagnostic, not a full-train BPB. Dropout disabled for both splits. No test scoring.",
        "sampled_train": training, "validation": validation,
        "validation_bpb": validation["all"]["nll_nats"] / math.log(2) / data["validation"][1],
        "validation_minus_train_mean_nll": validation["all"]["mean_nll_nats"] - training["all"]["mean_nll_nats"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
