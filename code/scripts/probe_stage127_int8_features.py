"""Input-only Stage92 CPU dynamic-INT8 feature diagnostic; no score/export."""
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

from common import load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from train_experiment import atomic_json_dump


def timed_features(model: nn.Module, ids: torch.Tensor, repeats: int = 8) -> list[float]:
    with torch.inference_mode():
        for _ in range(2):
            model.features(ids)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model.features(ids)
            samples.append(time.perf_counter() - started)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage127 evidence path")
    if sha(args.neural) != NEURAL_SHA:
        raise ValueError("Unexpected Stage92 neural checkpoint")
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    device, precision = setup("cpu", "fp32", 4)
    reference, _ = make_model(payload["implementation"], payload["config"], device)
    reference.load_state_dict(payload["model"], strict=True)
    reference.eval()

    ffn_only = copy.deepcopy(reference)
    for block in ffn_only.blocks:
        block.mlp = torch.ao.quantization.quantize_dynamic(
            block.mlp, {nn.Linear}, dtype=torch.qint8, inplace=False)
    all_linear = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(reference), {nn.Linear}, dtype=torch.qint8, inplace=False)
    variants = {"eager_fp32": reference, "ffn_int8": ffn_only,
                "all_linear_int8": all_linear}
    batches = windows(load_data()["validation"][0], 32)
    first, _ = next(batches)
    second, _ = next(batches)

    errors = {}
    with torch.inference_mode():
        for label, ids in (("first", first), ("second", second)):
            expected = reference.features(ids)
            expected_rms = expected.square().mean().sqrt()
            errors[label] = {}
            for name, model in variants.items():
                actual = model.features(ids)
                if actual.shape != expected.shape or not torch.isfinite(actual).all():
                    raise ValueError(f"Invalid {name} hidden features for {label}")
                delta = actual - expected
                errors[label][name] = {
                    "max_abs": float(delta.abs().max()),
                    "relative_rms": float(delta.square().mean().sqrt() / expected_rms),
                }
    samples = {name: timed_features(model, first) for name, model in variants.items()}
    medians = {name: statistics.median(values) for name, values in samples.items()}
    passes = {}
    for name in ("ffn_int8", "all_linear_int8"):
        passes[name] = (
            medians[name] <= .88 * medians["eager_fp32"]
            and max(errors[label][name]["relative_rms"] for label in errors) < .05)
    result = {
        "purpose": "stage127_int8_input_only_diagnostic_not_rule_qualified",
        "checkpoint_sha256": NEURAL_SHA,
        "source_sha256": sha(Path(__file__)),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "quantized_engine": torch.backends.quantized.engine,
        "input_shape": list(first.shape),
        "precision_of_eager_reference": precision,
        "errors": errors,
        "seconds_runs": samples,
        "median_seconds": medians,
        "passes_diagnostic_speed_and_error_gate": passes,
        "quantized_internal_arithmetic_requires_course_rule_clarification": True,
        "validation_inputs_without_labels": True,
        "full_validation_not_scored": True,
        "no_test_scoring": True,
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
