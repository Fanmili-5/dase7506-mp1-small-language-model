"""Validation diagnostic for exact within-window successor n-gram caching."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from train_experiment import atomic_json_dump

MAX_ORDER = 8
MIN_ORDERS = (1, 2, 3, 4, 5, 6)
WEIGHTS = (0., .05, .10, .20, .30, .50, .75, 1.)


def local_successor_statistics(ids, targets, max_order=MAX_ORDER):
    """Return fixed cache target probability and longest available order."""
    if ids.ndim != 1 or targets.shape != ids.shape:
        raise ValueError("Expected matching one-dimensional rows")
    tables = [None] + [defaultdict(Counter) for _ in range(max_order)]
    probability = torch.zeros(len(ids), dtype=torch.float64)
    matched_order = torch.zeros(len(ids), dtype=torch.int64)
    valid_length = int(targets.ne(-100).sum())
    values = ids.tolist(); target_values = targets.tolist()
    for position in range(valid_length):
        if position:
            predecessor = position - 1
            for order in range(1, min(max_order, position) + 1):
                context = tuple(values[predecessor - order + 1:predecessor + 1])
                tables[order][context][values[position]] += 1
        for order in range(min(max_order, position + 1), 0, -1):
            context = tuple(values[position - order + 1:position + 1])
            counts = tables[order].get(context)
            if counts:
                matched_order[position] = order
                probability[position] = counts[target_values[position]] / sum(counts.values())
                break
    return probability, matched_order


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
    tokens, byte_count = load_data()["validation"]
    totals = {(minimum, weight): 0. for minimum in MIN_ORDERS for weight in WEIGHTS}
    coverage = {minimum: 0 for minimum in MIN_ORDERS}
    target_count = 0; started = time.perf_counter()
    with torch.inference_mode():
        for ids, targets in windows(tokens, 32):
            gpu_ids = ids.to(device)
            safe = targets.clamp_min(0).to(device)
            base = model.predict_log_probs(gpu_ids).gather(-1, safe.unsqueeze(-1)).squeeze(-1).exp().cpu()
            for row in range(len(ids)):
                cache, orders = local_successor_statistics(ids[row], targets[row])
                valid = targets[row].ne(-100)
                base_row = base[row].double()
                for minimum in MIN_ORDERS:
                    available = orders.ge(minimum) & valid
                    coverage[minimum] += int(available.sum())
                    for weight in WEIGHTS:
                        probability = base_row.clone()
                        probability[available] = ((1 - weight) * base_row[available]
                                                  + weight * cache[available])
                        selected = probability[valid]
                        if (selected <= 0).any():
                            totals[(minimum, weight)] = float("inf")
                        elif math.isfinite(totals[(minimum, weight)]):
                            totals[(minimum, weight)] += float(-selected.log().sum())
                target_count += int(valid.sum())
    candidates = [dict(min_order=minimum, weight=weight, coverage=coverage[minimum],
                       coverage_fraction=coverage[minimum] / target_count,
                       nll_nats=nll, bpb=nll / math.log(2) / byte_count)
                  for (minimum, weight), nll in totals.items()]
    candidates.sort(key=lambda row: row["bpb"])
    result = dict(
        protocol=PROTOCOL, split="validation", status="diagnostic_completed",
        checkpoint_sha256=sha(args.checkpoint), implementation_sha256=implementation_sha,
        max_order=MAX_ORDER, grid=dict(min_orders=list(MIN_ORDERS), weights=list(WEIGHTS)),
        coverage=coverage, best=candidates[0], candidates=candidates,
        targets=target_count, utf8_bytes=byte_count, seconds=time.perf_counter() - started,
        target_only_scoring_of_fixed_normalized_expert=True,
        no_validation_gradient_updates=True, no_test_scoring=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"candidates": candidates[:12]}, indent=2))


if __name__ == "__main__":
    main()
