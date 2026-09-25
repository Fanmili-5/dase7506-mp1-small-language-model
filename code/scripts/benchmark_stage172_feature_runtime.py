"""Predeclared Linux CPU backend comparison on the frozen Stage143 ONNX graph.

This is a feature-only validation preflight. It cannot qualify a final model.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import openvino as ov
import torch

from common import PROTOCOL, load_data, setup, sha, windows


GRAPH = ROOT / "inference_assets/stage143-stage92-features.onnx"
GRAPH_SHA256 = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
CHECKPOINT_SHA256 = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.batches < 1:
        parser.error("Use a new output path and at least one batch")
    if sha(GRAPH) != GRAPH_SHA256 or sha(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Frozen Stage143 inputs changed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint["protocol"] != PROTOCOL
            or checkpoint["implementation"] != "student_stage143_openvino_singlepass"):
        raise ValueError("Unexpected checkpoint contract")
    setup("cpu", "fp32", 1)
    inputs = [x.numpy() for x, _ in itertools.islice(
        windows(load_data()["validation"][0], batch_size=32), args.batches)]
    if len(inputs) != args.batches:
        raise ValueError("Too few validation batches")

    core = ov.Core()
    openvino = core.compile_model(str(GRAPH), "CPU", {
        "INFERENCE_PRECISION_HINT": ov.Type.f32,
        "INFERENCE_NUM_THREADS": 1,
        "NUM_STREAMS": 1,
        "ENABLE_CPU_PINNING": False,
    })
    if (openvino.get_property("INFERENCE_PRECISION_HINT") != ov.Type.f32
            or int(openvino.get_property("INFERENCE_NUM_THREADS")) != 1
            or int(openvino.get_property("NUM_STREAMS")) != 1
            or openvino.get_property("ENABLE_CPU_PINNING")):
        raise ValueError("OpenVINO contract changed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnxruntime = ort.InferenceSession(str(GRAPH), sess_options=options,
                                       providers=["CPUExecutionProvider"])
    if onnxruntime.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("Unexpected ONNX Runtime provider")
    if len(onnxruntime.get_inputs()) != 1 or len(onnxruntime.get_outputs()) != 1:
        raise ValueError("Unexpected ONNX graph interface")
    input_name = onnxruntime.get_inputs()[0].name

    def run_openvino(x: np.ndarray) -> np.ndarray:
        return next(iter(openvino({"ids": x}).values()))

    def run_onnxruntime(x: np.ndarray) -> np.ndarray:
        return onnxruntime.run(None, {input_name: x})[0]

    runners = {"openvino": run_openvino, "onnxruntime": run_onnxruntime}
    for runner in runners.values():
        runner(inputs[0])
    observations = []
    sums = {name: [] for name in runners}
    max_abs_diff = 0.0
    all_finite = True
    # Paired batches, with reversed backend order in the second repetition.
    for repeat in range(2):
        elapsed_sums = {name: 0.0 for name in runners}
        for index, x in enumerate(inputs):
            outputs = {}
            order = ("openvino", "onnxruntime") if (repeat + index) % 2 == 0 else ("onnxruntime", "openvino")
            for name in order:
                start = time.perf_counter()
                outputs[name] = runners[name](x)
                elapsed = time.perf_counter() - start
                elapsed_sums[name] += elapsed
                observations.append({"repeat": repeat, "batch": index,
                                     "backend": name, "seconds": elapsed})
            a, b = outputs["openvino"], outputs["onnxruntime"]
            if a.shape != (*x.shape, 288) or b.shape != a.shape or a.dtype != np.float32 or b.dtype != np.float32:
                raise ValueError("Backend output shape or dtype mismatch")
            all_finite &= bool(np.isfinite(a).all() and np.isfinite(b).all())
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(a - b))))
        for name in sums:
            sums[name].append(elapsed_sums[name])
    medians = {name: statistics.median(values) for name, values in sums.items()}
    speedup_fraction = 1.0 - medians["onnxruntime"] / medians["openvino"]
    parity_pass = all_finite and max_abs_diff <= 1e-3
    speed_gate_pass = speedup_fraction >= 0.12
    result = dict(
        protocol=PROTOCOL, purpose="stage172_linux_feature_backend_preflight",
        split="validation", device="cpu", precision="fp32", threads=1,
        platform=platform.platform(), torch_version=torch.__version__,
        openvino_version=ov.__version__, onnxruntime_version=ort.__version__,
        checkpoint_sha256=sha(args.checkpoint), graph_sha256=sha(GRAPH),
        benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        batches=args.batches, repetitions=2, backend_order="alternating_by_batch_and_repeat",
        seconds_by_backend=sums, median_seconds=medians, observations=observations,
        max_abs_hidden_difference=max_abs_diff, all_outputs_finite=all_finite,
        hidden_parity_threshold=1e-3, hidden_parity_pass=parity_pass,
        required_feature_speedup_fraction=0.12, observed_feature_speedup_fraction=speedup_fraction,
        feature_speed_gate_pass=speed_gate_pass,
        proceed_to_full_validation_candidate=parity_pass and speed_gate_pass,
        no_test_scoring=True,
        warning="Feature-only preflight; not a complete score or resource qualification",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "median_seconds", "max_abs_hidden_difference", "hidden_parity_pass",
        "observed_feature_speedup_fraction", "feature_speed_gate_pass",
        "proceed_to_full_validation_candidate")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
