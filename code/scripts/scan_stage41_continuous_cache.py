"""Validation-only diagnostic for a fixed causal continuous successor cache."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from train_experiment import atomic_json_dump

SCALES = (2., 4., 8., 16., 32., 64.)
WEIGHTS = (0., .025, .05, .075, .10, .15, .20, .30)


def successor_target_probability(hidden, ids, targets, scale):
    """Probability assigned by fixed cosine cache to each supplied target.

    Targets are used only to score a fixed normalized distribution efficiently;
    they do not influence attention scores, available values, or model behavior.
    """
    batch, length, _ = hidden.shape
    normalized = F.normalize(hidden.float(), dim=-1)
    scores = (normalized @ normalized.transpose(-1, -2)) * float(scale)
    allowed = torch.ones(length, length, device=ids.device, dtype=torch.bool).tril(-1)
    allowed[0, 0] = True
    attention = scores.masked_fill(~allowed, -float("inf")).softmax(-1)
    attention = attention * (torch.arange(length, device=ids.device) > 0)[None, :, None]
    values = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
    safe_targets = targets.clamp_min(0)
    matches = values[:, None, :].eq(safe_targets[:, :, None])
    probability = (attention * matches).sum(-1)
    if probability.shape != (batch, length):
        raise AssertionError("Unexpected cache probability shape")
    return probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device, _ = setup("cuda", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL:
        raise ValueError("Wrong protocol")
    model, implementation_sha = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"]); model.eval()
    data = load_data()
    tokens, byte_count = data["validation"]
    totals = {(scale, weight): 0. for scale in SCALES for weight in WEIGHTS}
    target_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for ids, targets in windows(tokens, 32):
            ids, targets = ids.to(device), targets.to(device)
            base_logp = model.predict_log_probs(ids).float()
            hidden = model.features(ids).float()
            safe = targets.clamp_min(0)
            base_probability = base_logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1).exp()
            valid = targets.ne(-100)
            cache_available = (torch.arange(ids.shape[1], device=device) > 0)[None, :]
            for scale in SCALES:
                cache_probability = successor_target_probability(hidden, ids, targets, scale)
                for weight in WEIGHTS:
                    effective = cache_available * weight
                    probability = base_probability * (1 - effective) + cache_probability * effective
                    totals[(scale, weight)] += float(-probability[valid].double().log().sum())
            target_count += int(valid.sum())
    candidates = [dict(scale=scale, weight=weight, nll_nats=nll,
                       bpb=nll / math.log(2) / byte_count)
                  for (scale, weight), nll in totals.items()]
    candidates.sort(key=lambda row: row["bpb"])
    result = dict(
        protocol=PROTOCOL, split="validation", status="diagnostic_completed",
        checkpoint_sha256=sha(args.checkpoint), implementation_sha256=implementation_sha,
        grid=dict(scales=list(SCALES), weights=list(WEIGHTS)),
        best=candidates[0], candidates=candidates, targets=target_count,
        utf8_bytes=byte_count, seconds=time.perf_counter() - started,
        target_only_scoring_of_fixed_normalized_expert=True,
        no_validation_gradient_updates=True, no_test_scoring=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"candidates": candidates[:12]}, indent=2))


if __name__ == "__main__":
    main()
