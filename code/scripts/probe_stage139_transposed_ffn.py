"""Pilot frozen SwiGLU FP32 GEMM with contiguous transposed weights."""
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

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from train_experiment import atomic_json_dump


class TransposedLinear(nn.Module):
    def __init__(self, source: nn.Linear):
        super().__init__()
        self.weight_t = nn.Parameter(source.weight.detach().T.contiguous(),
                                     requires_grad=False)
        self.bias = (nn.Parameter(source.bias.detach().clone(), requires_grad=False)
                     if source.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if self.bias is None:
            output = flat @ self.weight_t
        else:
            output = torch.addmm(self.bias, flat, self.weight_t)
        return output.reshape(*x.shape[:-1], self.weight_t.shape[-1])


def replace_ffns(model: nn.Module, selection: str) -> int:
    replaced = 0
    for block in model.blocks:
        for name in ("input", "output"):
            if selection not in (name, "both"):
                continue
            original = getattr(block.mlp, name)
            if not isinstance(original, nn.Linear):
                raise TypeError("Unexpected SwiGLU projection")
            setattr(block.mlp, name, TransposedLinear(original))
            replaced += 1
    if replaced != (16 if selection == "both" else 8):
        raise ValueError("Unexpected FFN count")
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage139 evidence")
    if sha(args.neural) != NEURAL_SHA:
        raise ValueError("Unexpected frozen Stage92 neural checkpoint")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected protocol")
    eager, _ = make_model(payload["implementation"], payload["config"], device)
    eager.load_state_dict(payload["model"], strict=True)
    eager.eval()
    source = windows(load_data()["validation"][0], 32)
    first, _ = next(source)
    second, _ = next(source)
    variants = {"eager": eager}
    layer_counts = {}
    for selection in ("input", "output", "both"):
        candidate = copy.deepcopy(eager).eval()
        layer_counts[selection] = replace_ffns(candidate, selection)
        variants[selection] = candidate
    with torch.inference_mode():
        reference = [eager.features(ids) for ids in (first, second)]
        errors = {}
        for selection, candidate in variants.items():
            if selection == "eager":
                continue
            actual = [candidate.features(ids) for ids in (first, second)]
            if not all(torch.isfinite(x).all() for x in actual):
                raise ValueError("Nonfinite features: " + selection)
            errors[selection] = max(float((a - b).abs().max())
                                    for a, b in zip(actual, reference))
        for model in variants.values():
            for _ in range(2):
                model.features(first)
        samples = {name: [] for name in variants}
        names = tuple(variants)
        for round_index in range(8):
            for offset in range(len(names)):
                name = names[(round_index + offset) % len(names)]
                started = time.perf_counter()
                variants[name].features(first)
                samples[name].append(time.perf_counter() - started)
    medians = {name: statistics.median(values)
               for name, values in samples.items()}
    passing = {name: errors[name] <= 3e-5
               and medians[name] <= .92 * medians["eager"]
               for name in errors}
    result = dict(
        protocol=PROTOCOL, split="validation_inputs_only", precision="fp32",
        stage92_neural_sha256=NEURAL_SHA, source_sha256=sha(Path(__file__)),
        torch_version=torch.__version__, platform=platform.platform(),
        threads=torch.get_num_threads(), input_shapes=[list(first.shape),
                                                        list(second.shape)],
        transposed_ffn_linears=layer_counts, max_feature_abs_error=errors,
        seconds_runs=samples, median_seconds=medians,
        relative_time={name: medians[name] / medians["eager"]
                       for name in errors},
        meets_8_percent_speed_and_parity_gate=passing,
        no_validation_labels_used=True, no_test_scoring=True,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
