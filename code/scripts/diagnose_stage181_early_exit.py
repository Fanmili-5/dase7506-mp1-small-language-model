"""Frozen, target-only validation pilot for untrained intermediate-layer readouts.

This is a diagnostic, not an inference candidate: it reuses the final output
head on blocks 4 and 6 and never fits parameters or chooses a target-aware gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows

LAYERS = (4, 6)
WEIGHTS = (0.05, 0.10, 0.20)


def validation_tokens() -> tuple[torch.Tensor, int, str]:
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    name = "wikitext_validation.txt"
    path = ROOT / "data" / name
    if sha(path) != manifest["sha256"][name]:
        raise ValueError("Validation file changed")
    tokenizer_path = ROOT / "data/tokenizer.json"
    if sha(tokenizer_path) != manifest["sha256"]["tokenizer.json"]:
        raise ValueError("Tokenizer changed")
    raw = path.read_bytes()
    ids = Tokenizer.from_file(str(tokenizer_path)).encode(raw.decode("utf-8")).ids
    return torch.tensor(ids, dtype=torch.long), len(raw), sha(path)


@torch.inference_mode()
def pilot(model, tokens: torch.Tensor, device: torch.device, max_batches: int) -> dict:
    captured: dict[int, torch.Tensor] = {}
    hooks = [model.blocks[layer - 1].register_forward_hook(
        lambda _module, _inputs, output, layer=layer: captured.__setitem__(layer, output)
    ) for layer in LAYERS]
    names = ["final"] + [f"layer{layer}_alone" for layer in LAYERS]
    names += [f"layer{layer}_weight{weight}" for layer in LAYERS for weight in WEIGHTS]
    nll = {name: 0.0 for name in names}
    target_count = 0
    try:
        for batch_idx, (x, y) in enumerate(windows(tokens, 32)):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            valid = y != -100
            targets = y.clamp_min(0).unsqueeze(-1)
            captured.clear()
            final = model.predict_log_probs(x).float().gather(-1, targets).squeeze(-1)
            if set(captured) != set(LAYERS):
                raise ValueError("Missing intermediate activation")
            nll["final"] += float(-final[valid].double().sum())
            target_count += int(valid.sum())
            for layer in LAYERS:
                hidden = model.norm(captured[layer]).float()
                early = F.log_softmax(model.head(hidden) + model.output_bias, dim=-1)
                early = early.gather(-1, targets).squeeze(-1)
                nll[f"layer{layer}_alone"] += float(-early[valid].double().sum())
                for weight in WEIGHTS:
                    mixed = torch.logaddexp(
                        final + math.log1p(-weight), early + math.log(weight)
                    )
                    nll[f"layer{layer}_weight{weight}"] += float(-mixed[valid].double().sum())
    finally:
        for hook in hooks:
            hook.remove()
    if target_count == 0:
        raise ValueError("No scored targets")
    reference = nll["final"] / target_count
    return {
        "batches": min(batch_idx + 1, max_batches),
        "targets": target_count,
        "mean_nll_nats": {name: value / target_count for name, value in nll.items()},
        "delta_bits_per_target_vs_final": {
            name: (value / target_count - reference) / math.log(2)
            for name, value in nll.items() if name != "final"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.max_batches <= 0:
        parser.error("Output must be new and max-batches positive")
    device, _ = setup(args.device, "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol") != PROTOCOL or checkpoint.get("implementation") != "student_hybrid_conv_output_bias":
        raise ValueError("Unexpected checkpoint")
    model, implementation_sha = make_model(checkpoint["implementation"], checkpoint["config"], device)
    if len(model.blocks) < max(LAYERS):
        raise ValueError("Model too shallow")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    tokens, validation_bytes, validation_sha = validation_tokens()
    metrics = pilot(model, tokens, device, args.max_batches)
    result = {
        "protocol": PROTOCOL,
        "split": "validation_prefix_only",
        "precision": "fp32",
        "device": str(device),
        "checkpoint_sha256": sha(args.checkpoint),
        "implementation_sha256": implementation_sha,
        "validation_sha256": validation_sha,
        "validation_bytes_total": validation_bytes,
        "predeclared_layers": LAYERS,
        "predeclared_weights": WEIGHTS,
        "pass_gate": "At least 0.02 bits per target improvement on this pilot before full validation; resource compliance is a separate gate.",
        "no_test_scoring": True,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
