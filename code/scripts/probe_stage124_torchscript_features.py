"""Compare eager and TorchScript CPU feature extraction without using labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from common import load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA


class FeatureWrapper(nn.Module):
    def __init__(self, neural: nn.Module):
        super().__init__()
        self.neural = neural

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.neural.features(ids)


def samples(model, ids, repeats=6):
    with torch.inference_mode():
        for _ in range(2):
            model(ids)
        rows = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(ids)
            rows.append(time.perf_counter() - started)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage124 output path")
    if sha(args.neural) != NEURAL_SHA:
        raise ValueError("Unexpected Stage92 neural checkpoint")
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(payload["implementation"], payload["config"], device)
    neural.load_state_dict(payload["model"], strict=True)
    neural.eval()
    eager = FeatureWrapper(neural).eval()
    inputs = windows(load_data()["validation"][0], 32)
    first, _ = next(inputs)
    second, _ = next(inputs)
    with torch.inference_mode():
        traced = torch.jit.trace(eager, first, check_trace=False)
        frozen = torch.jit.freeze(traced)
        optimized = torch.jit.optimize_for_inference(frozen)
        reference_first = eager(first)
        reference_second = eager(second)
        errors = {}
        for name, model in (("frozen", frozen), ("optimized", optimized)):
            actual_first = model(first)
            actual_second = model(second)
            errors[name] = max(
                float((actual_first - reference_first).abs().max()),
                float((actual_second - reference_second).abs().max()))
            if not torch.isfinite(actual_first).all() or not torch.isfinite(actual_second).all():
                raise ValueError(f"Nonfinite Stage124 {name} output")
    results = {name: samples(model, first)
               for name, model in (("eager", eager), ("frozen", frozen),
                                   ("optimized", optimized))}
    medians = {name: statistics.median(times) for name, times in results.items()}
    result = dict(stage92_neural_sha256=NEURAL_SHA,
                  source_sha256=sha(Path(__file__)),
                  torch_version=torch.__version__,
                  thread_count=torch.get_num_threads(),
                  input_shape=list(first.shape),
                  second_input_shape=list(second.shape),
                  max_hidden_error=errors,
                  seconds_runs=results, median_seconds=medians,
                  optimized_relative_time=medians["optimized"] / medians["eager"],
                  meets_12_percent_feature_speed_gate=(
                      medians["optimized"] <= .88 * medians["eager"]
                      and errors["optimized"] <= 3e-5),
                  validation_inputs_without_labels=True, no_test_scoring=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
