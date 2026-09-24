"""Compare exact Stage115 sparse-addition kernels on one fixed input batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import load_data, make_model, setup, sha, windows


STAGE115_SHA = "902e4b21c9ddf3afda2258ae032fe852517716cccfd5341567b2fabc21910e76"


def add_index_add(self, result, weight, terms, suffix):
    flat = result.view(-1, self.vocab)
    flat.addcmul_((suffix * weight.flatten())[:, None], self.unigram[None, :])
    flat1 = flat.view(-1)
    flat_weight = weight.flatten()
    for rows, columns, mass in reversed(terms):
        flat1.index_add_(0, rows * self.vocab + columns,
                         mass * flat_weight[rows])
    return result


def add_scatter_add(self, result, weight, terms, suffix):
    flat = result.view(-1, self.vocab)
    flat.addcmul_((suffix * weight.flatten())[:, None], self.unigram[None, :])
    flat1 = flat.view(-1)
    flat_weight = weight.flatten()
    for rows, columns, mass in reversed(terms):
        flat1.scatter_add_(0, rows * self.vocab + columns,
                           mass * flat_weight[rows])
    return result


def time_model(model, ids, repeats=10):
    with torch.inference_mode():
        for _ in range(2):
            model.predict_log_probs(ids)
        samples = []
        for _ in range(repeats):
            before = time.perf_counter()
            model.predict_log_probs(ids)
            samples.append(time.perf_counter() - before)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha(args.checkpoint) != STAGE115_SHA:
        raise ValueError("Use the pinned Stage115 checkpoint and a new output")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device, _ = setup("cpu", "fp32", 4)
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    ids, _ = next(windows(load_data()["validation"][0], 32))
    original = model.ngram.add_collected
    with torch.inference_mode():
        reference = model.predict_log_probs(ids)
    results = {}
    for name, addition in (("advanced_index", original),
                           ("index_add", MethodType(add_index_add, model.ngram)),
                           ("scatter_add", MethodType(add_scatter_add, model.ngram))):
        model.ngram.add_collected = addition
        with torch.inference_mode():
            actual = model.predict_log_probs(ids)
        probability_error = float((actual.exp() - reference.exp()).abs().max())
        logp_error = float((actual - reference).abs().max())
        if probability_error > 3e-6 or logp_error > 3e-5:
            raise ValueError(f"{name} changed Stage115 predictions")
        samples = time_model(model, ids)
        results[name] = dict(probability_error=probability_error,
                             logp_error=logp_error, seconds=samples,
                             median=statistics.median(samples))
        print(json.dumps(dict(kernel=name, **results[name])), flush=True)
    model.ngram.add_collected = original
    result = dict(stage115_sha256=STAGE115_SHA, kernels=results,
                  source_sha256=sha(Path(__file__)), no_test_scoring=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result | {"kernels": {}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
