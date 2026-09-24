"""Export Stage92 neural features and compare ORT versus eager CPU, label-free."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import onnxruntime as ort
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


def benchmark(fn, ids, repeats=6):
    for _ in range(2):
        fn(ids)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(ids)
        samples.append(time.perf_counter() - started)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage126 run directory")
    if sha(args.neural) != NEURAL_SHA:
        raise ValueError("Unexpected Stage92 neural checkpoint")
    args.run_dir.mkdir(parents=True)
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(payload["implementation"], payload["config"], device)
    neural.load_state_dict(payload["model"], strict=True)
    neural.eval()
    eager = FeatureWrapper(neural).eval()
    source = windows(load_data()["validation"][0], 32)
    first, _ = next(source)
    second, _ = next(source)
    graph = args.run_dir / "stage126-features.onnx"
    export_started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            eager, first, str(graph), export_params=True,
            opset_version=17, do_constant_folding=True,
            input_names=["ids"], output_names=["hidden"],
            dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}})
    export_seconds = time.perf_counter() - export_started
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_started = time.perf_counter()
    session = ort.InferenceSession(str(graph), options,
                                   providers=["CPUExecutionProvider"])
    session_creation_seconds = time.perf_counter() - session_started

    def run_ort(ids):
        output = session.run(["hidden"], {"ids": ids.numpy()})[0]
        return torch.from_numpy(output)

    def run_eager(ids):
        with torch.inference_mode():
            return eager(ids)

    errors = {}
    for label, ids in (("first", first), ("second", second), ("single", first[:1])):
        expected = run_eager(ids)
        actual = run_ort(ids)
        if actual.shape != expected.shape or not torch.isfinite(actual).all():
            raise ValueError(f"Stage126 ORT output invalid for {label}")
        errors[label] = float((actual - expected).abs().max())
    samples = {"eager": benchmark(run_eager, first),
               "onnxruntime": benchmark(run_ort, first)}
    medians = {name: statistics.median(rows) for name, rows in samples.items()}
    result = dict(stage92_neural_sha256=NEURAL_SHA,
                  source_sha256=sha(Path(__file__)),
                  torch_version=torch.__version__,
                  onnxruntime_version=ort.__version__,
                  input_shape=list(first.shape),
                  graph_sha256=sha(graph), graph_bytes=graph.stat().st_size,
                  export_seconds=export_seconds,
                  session_creation_seconds=session_creation_seconds,
                  max_hidden_error=errors, seconds_runs=samples,
                  median_seconds=medians,
                  ort_relative_time=medians["onnxruntime"] / medians["eager"],
                  meets_12_percent_feature_speed_gate=(
                      max(errors.values()) <= 3e-4
                      and medians["onnxruntime"] <= .88 * medians["eager"]),
                  validation_inputs_without_labels=True,
                  no_test_scoring=True)
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
