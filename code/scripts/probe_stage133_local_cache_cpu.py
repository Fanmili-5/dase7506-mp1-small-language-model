"""One-batch CPU FP32 parity and timing pilot for deployable local cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.scan_stage43_exact_local_cache import local_successor_statistics
from scripts.train_experiment import atomic_json_dump
from student_stage133_exact_local_cache import add_exact_local_cache

STAGE115_SHA = "902e4b21c9ddf3afda2258ae032fe852517716cccfd5341567b2fabc21910e76"


def timings(model, ids, repeats=6):
    samples = []
    with torch.no_grad():
        for _ in range(2):
            model.predict_log_probs(ids)
        for _ in range(repeats):
            started = time.perf_counter()
            model.predict_log_probs(ids)
            samples.append(time.perf_counter() - started)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.checkpoint) != STAGE115_SHA:
        raise ValueError("Stage115 checkpoint hash mismatch")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_stage115_order5_gate":
        raise ValueError("Wrong source checkpoint")
    base, _ = make_model(payload["implementation"], payload["config"], device)
    candidate, candidate_sha = make_model("student_stage133_exact_local_cache", payload["config"], device)
    base.load_state_dict(payload["model"])
    candidate.load_state_dict(payload["model"])
    base.eval(); candidate.eval()
    ids, targets = next(windows(load_data()["validation"][0], 32))
    with torch.no_grad():
        base_logp = base.predict_log_probs(ids)
        candidate_logp = candidate.predict_log_probs(ids)
        expected = add_exact_local_cache(base_logp.exp(), ids).log()
        max_log_difference = float((candidate_logp - expected).abs().max())
        max_log_norm = float(torch.logsumexp(candidate_logp, dim=-1).abs().max())
        target_errors = []
        for row in range(ids.shape[0]):
            cache, orders = local_successor_statistics(ids[row], targets[row])
            for position in range(ids.shape[1]):
                if targets[row, position] == -100:
                    continue
                token = int(targets[row, position])
                expected_target = float(base_logp[row, position, token].exp())
                if orders[position] >= 2:
                    expected_target = 0.9 * expected_target + 0.1 * float(cache[position])
                target_errors.append(abs(float(candidate_logp[row, position, token].exp()) - expected_target))
    base_samples = timings(base, ids)
    candidate_samples = timings(candidate, ids)
    base_median = sorted(base_samples)[len(base_samples) // 2 - 1:len(base_samples) // 2 + 1]
    candidate_median = sorted(candidate_samples)[len(candidate_samples) // 2 - 1:len(candidate_samples) // 2 + 1]
    base_median = sum(base_median) / 2
    candidate_median = sum(candidate_median) / 2
    result = dict(protocol=PROTOCOL, checkpoint_sha256=STAGE115_SHA,
                  implementation_sha256=candidate_sha, split="validation_inputs_only",
                  no_test_scoring=True, source_sha256=sha(Path(__file__)),
                  batch_shape=list(ids.shape), max_log_difference=max_log_difference,
                  max_log_normalization_error=max_log_norm,
                  max_target_probability_error=max(target_errors),
                  base_seconds=base_samples, candidate_seconds=candidate_samples,
                  base_median_seconds=base_median, candidate_median_seconds=candidate_median,
                  candidate_to_base_time_ratio=candidate_median / base_median)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))
    if max_log_difference > 3e-6 or max_log_norm > 1e-5 or max(target_errors) > 3e-6:
        raise ValueError("Local cache parity/normalization failed")


if __name__ == "__main__":
    main()
