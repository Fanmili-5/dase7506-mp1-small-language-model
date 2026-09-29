"""Train-prefix-only SwiGLU channel-importance and perturbation probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import load_data, make_model, setup, sha


STAGE105_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"
RATIOS = (.10, .20, .25, .30)


def sampled_windows(tokens: torch.Tensor, seed: int, windows: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(len(tokens) - 257, (windows,), generator=generator)
    return tokens[starts[:, None] + torch.arange(256)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.checkpoint) != STAGE105_SHA or args.output.exists():
        raise ValueError("Unexpected source checkpoint or existing output")
    device, _ = setup("cuda", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    tokens = load_data()["train"][0]
    blocks = list(model.neural.blocks)
    stats = [torch.zeros(block.mlp.output.in_features, device=device)
             for block in blocks]
    handles = []
    for index, block in enumerate(blocks):
        def accumulate(module, inputs, i=index):
            stats[i].add_(inputs[0].float().square().sum((0, 1)))
        handles.append(block.mlp.output.register_forward_pre_hook(accumulate))
    sample = sampled_windows(tokens, 108017, 96)
    with torch.inference_mode():
        for chunk in sample.split(16):
            model.neural.features(chunk.to(device))
    for handle in handles:
        handle.remove()
    importance = []
    for index, block in enumerate(blocks):
        output_norm = block.mlp.output.weight.detach().float().square().sum(0)
        importance.append(stats[index] * output_norm)
    rows = []
    for index, scores in enumerate(importance):
        ordered = scores.sort().values
        rows.append({
            "block": index + 1,
            "channels": int(scores.numel()),
            "relative_proxy_energy_removed": {
                str(ratio): float(ordered[:round(ratio * len(ordered))].sum()
                                  / ordered.sum()) for ratio in RATIOS
            },
        })
    probe = sampled_windows(tokens, 108018, 16).to(device)
    with torch.inference_mode():
        original = model.neural.features(probe)
    perturbations = {}
    for ratio in RATIOS:
        handles = []
        for index, block in enumerate(blocks):
            keep = torch.ones_like(importance[index])
            lowest = importance[index].argsort()[:round(ratio * keep.numel())]
            keep[lowest] = 0
            def mask(module, inputs, vector=keep):
                return (inputs[0] * vector,)
            handles.append(block.mlp.output.register_forward_pre_hook(mask))
        with torch.inference_mode():
            changed = model.neural.features(probe)
        for handle in handles:
            handle.remove()
        diff = changed - original
        perturbations[str(ratio)] = {
            "feature_rms_ratio": float(diff.square().mean().sqrt()
                                       / original.square().mean().sqrt()),
            "feature_max_absolute_error": float(diff.abs().max()),
        }
    result = {
        "purpose": "train_prefix_channel_importance_no_validation_or_test_scoring",
        "source_checkpoint_sha256": STAGE105_SHA,
        "importance_windows": 96,
        "perturbation_windows": 16,
        "ratios": RATIOS,
        "layers": rows,
        "perturbations": perturbations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
