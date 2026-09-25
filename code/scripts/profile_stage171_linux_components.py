"""Time Stage143 feature, count and remaining inference work on validation inputs."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.batches < 1:
        parser.error("Use a new output and a positive number of batches")
    device, precision = setup("cpu", "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint["protocol"] != PROTOCOL
            or checkpoint["implementation"] != "student_stage143_openvino_singlepass"):
        raise ValueError("Unexpected checkpoint for Stage171 profile")
    model, implementation_sha = make_model(checkpoint["implementation"],
                                            checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    validation = load_data()["validation"]
    batches = list(itertools.islice(windows(validation[0], batch_size=32), args.batches))
    if len(batches) != args.batches:
        raise ValueError("Too few validation batches")
    times = {"features": 0.0, "count_collect": 0.0, "count_add": 0.0}
    calls = {name: 0 for name in times}

    def timed(target, label):
        def wrapper(*arguments, **keywords):
            started = time.perf_counter()
            value = target(*arguments, **keywords)
            times[label] += time.perf_counter() - started
            calls[label] += 1
            return value
        return wrapper

    model.neural.features = timed(model.neural.features, "features")
    model.ngram.collect = timed(model.ngram.collect, "count_collect")
    model.ngram.add_collected = timed(model.ngram.add_collected, "count_add")

    with torch.no_grad():
        # Warm compilation/cache effects outside the measured region.
        model.predict_log_probs(batches[0][0])
        for key in times:
            times[key] = 0.0
            calls[key] = 0
        inference_seconds = 0.0
        scorer_check_seconds = 0.0
        scored_targets = 0
        nll_nats = 0.0
        for x, y in batches:
            started = time.perf_counter()
            logp = model.predict_log_probs(x).float()
            inference_seconds += time.perf_counter() - started
            started = time.perf_counter()
            if logp.shape != (*x.shape, 2048) or not torch.isfinite(logp).all():
                raise ValueError("Invalid log-probability shape or values")
            if torch.logsumexp(logp, dim=-1).abs().max().item() > 1e-3:
                raise ValueError("Distribution is not normalized")
            losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            losses.masked_fill_(y == -100, 0)
            nll_nats += float(losses.double().sum())
            scored_targets += int((y != -100).sum())
            scorer_check_seconds += time.perf_counter() - started
    remainder = inference_seconds - sum(times.values())
    if remainder < -1e-6 or not math.isfinite(remainder):
        raise ValueError("Invalid component time decomposition")
    result = dict(
        protocol=PROTOCOL, purpose="linux_stage143_partial_validation_runtime_profile",
        split="validation", precision=precision, device=str(device),
        requested_threads=args.threads, platform=platform.platform(),
        torch_version=torch.__version__, checkpoint_sha256=sha(args.checkpoint),
        implementation_sha256=implementation_sha,
        graph_sha256=sha(ROOT / "inference_assets/stage143-stage92-features.onnx"),
        batches=args.batches, input_rows=args.batches * 32,
        scored_targets=scored_targets, sampled_nll_nats=nll_nats,
        inference_seconds=inference_seconds,
        scorer_check_seconds=scorer_check_seconds,
        timed_component_seconds=times, timed_component_calls=calls,
        uninstrumented_inference_seconds=remainder,
        no_test_scoring=True,
        warning="Timing instruments only the first validation batches, not the full resource gate",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
