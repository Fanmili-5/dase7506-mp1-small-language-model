"""Pilot exact FP32 oneDNN weight packing on frozen Stage92 FFNs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.mkldnn import MkldnnLinear

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from train_experiment import atomic_json_dump


def pack_ffns(model: nn.Module, selection: str) -> int:
    """Replace only selected frozen SwiGLU linears with FP32 packed weights."""
    count = 0
    for block in model.blocks:
        mlp = block.mlp
        if selection in {"input", "both"}:
            if not isinstance(mlp.input, nn.Linear):
                raise TypeError("Unexpected SwiGLU input projection")
            mlp.input = MkldnnLinear(mlp.input, torch.float32)
            count += 1
        if selection in {"output", "both"}:
            if not isinstance(mlp.output, nn.Linear):
                raise TypeError("Unexpected SwiGLU output projection")
            mlp.output = MkldnnLinear(mlp.output, torch.float32)
            count += 1
    if count != (16 if selection == "both" else 8):
        raise ValueError("Unexpected number of FFN projections")
    return count


def timings(model, ids, repeats=6):
    samples = []
    with torch.inference_mode():
        for _ in range(2):
            model.features(ids)
        for _ in range(repeats):
            started = time.perf_counter()
            model.features(ids)
            samples.append(time.perf_counter() - started)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.neural) != NEURAL_SHA:
        raise ValueError("Unexpected Stage92 neural checkpoint")
    device, _ = setup("cpu", "fp32", 4)
    if not torch.backends.mkldnn.is_available():
        raise RuntimeError("oneDNN unavailable on this host")
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL:
        raise ValueError("Wrong protocol")
    eager, _ = make_model(payload["implementation"], payload["config"], device)
    eager.load_state_dict(payload["model"], strict=True)
    eager.eval()
    source = windows(load_data()["validation"][0], 32)
    first, _ = next(source)
    second, _ = next(source)
    with torch.inference_mode():
        reference = (eager.features(first), eager.features(second))
    variants = {}
    errors = {}
    packed_layers = {}
    for selection in ("input", "output", "both"):
        candidate = copy.deepcopy(eager).eval()
        packed_layers[selection] = pack_ffns(candidate, selection)
        with torch.inference_mode():
            actual = (candidate.features(first), candidate.features(second))
            errors[selection] = max(float((a - b).abs().max())
                                    for a, b in zip(actual, reference))
            if not all(torch.isfinite(a).all() for a in actual):
                raise ValueError(f"Nonfinite {selection} features")
        variants[selection] = candidate
    samples = {"eager": timings(eager, first)}
    for selection, candidate in variants.items():
        samples[selection] = timings(candidate, first)
    medians = {name: statistics.median(times) for name, times in samples.items()}
    passing = {name: (errors[name] <= 3e-5
                      and medians[name] <= .88 * medians["eager"])
               for name in variants}
    result = dict(protocol=PROTOCOL, stage92_neural_sha256=NEURAL_SHA,
                  source_sha256=sha(Path(__file__)), split="validation_inputs_only",
                  no_test_scoring=True, torch_version=torch.__version__,
                  platform=platform.platform(), threads=torch.get_num_threads(),
                  mkldnn_available=torch.backends.mkldnn.is_available(),
                  input_shapes=[list(first.shape), list(second.shape)],
                  packed_ffn_linears=packed_layers, max_feature_abs_error=errors,
                  seconds_runs=samples, median_seconds=medians,
                  relative_time={name: medians[name] / medians["eager"]
                                 for name in variants},
                  meets_12_percent_speed_and_parity_gate=passing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
