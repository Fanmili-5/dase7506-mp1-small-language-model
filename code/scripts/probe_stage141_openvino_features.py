"""Compare frozen Stage92 FP32 feature inference on OpenVINO CPU and PyTorch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov
import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from train_experiment import atomic_json_dump


GRAPH_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage141 pilot")
    if sha(args.neural) != NEURAL_SHA or sha(args.graph) != GRAPH_SHA:
        raise ValueError("Unexpected frozen Stage92 neural/ONNX graph")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected source protocol")
    neural, _ = make_model(payload["implementation"], payload["config"], device)
    neural.load_state_dict(payload["model"], strict=True)
    neural.eval()
    source = windows(load_data()["validation"][0], 32)
    first, _ = next(source)
    second, _ = next(source)

    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise RuntimeError("OpenVINO CPU device unavailable")
    config = {
        "INFERENCE_PRECISION_HINT": ov.Type.f32,
        "INFERENCE_NUM_THREADS": 4,
        "NUM_STREAMS": 1,
        "ENABLE_CPU_PINNING": False,
    }
    compile_started = time.perf_counter()
    compiled = core.compile_model(str(args.graph), "CPU", config)
    compile_seconds = time.perf_counter() - compile_started
    precision = compiled.get_property("INFERENCE_PRECISION_HINT")
    threads = compiled.get_property("INFERENCE_NUM_THREADS")
    streams = compiled.get_property("NUM_STREAMS")
    pinning = compiled.get_property("ENABLE_CPU_PINNING")
    if precision != ov.Type.f32 or int(threads) != 4 or int(streams) != 1 or pinning:
        raise ValueError(f"OpenVINO resource/precision mismatch: {precision}, "
                         f"{threads}, {streams}, {pinning}")

    def eager(ids: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return neural.features(ids)

    def openvino(ids: torch.Tensor) -> torch.Tensor:
        output = compiled({"ids": ids.numpy()})
        return torch.from_numpy(next(iter(output.values())))

    errors = {}
    for label, ids in (("first", first), ("second", second),
                       ("single", first[:1])):
        expected = eager(ids)
        actual = openvino(ids)
        if actual.shape != expected.shape or not torch.isfinite(actual).all():
            raise ValueError(f"OpenVINO hidden output invalid: {label}")
        errors[label] = float((actual - expected).abs().max())
    funcs = {"eager": eager, "openvino": openvino}
    samples = {name: [] for name in funcs}
    for fn in funcs.values():
        for _ in range(2):
            fn(first)
    for round_index in range(8):
        for name in (("eager", "openvino") if round_index % 2 == 0
                     else ("openvino", "eager")):
            started = time.perf_counter()
            funcs[name](first)
            samples[name].append(time.perf_counter() - started)
    medians = {name: statistics.median(rows) for name, rows in samples.items()}
    passing = (max(errors.values()) <= 3e-4
               and medians["openvino"] <= .85 * medians["eager"])
    result = dict(
        protocol=PROTOCOL, split="validation_inputs_only", precision="fp32",
        stage92_neural_sha256=NEURAL_SHA, graph_sha256=GRAPH_SHA,
        source_sha256=sha(Path(__file__)),
        torch_version=torch.__version__, openvino_version=ov.__version__,
        platform=platform.platform(), torch_threads=torch.get_num_threads(),
        compiled_inference_precision=str(precision),
        compiled_inference_threads=int(threads),
        compiled_streams=int(streams), compiled_cpu_pinning=bool(pinning),
        input_shapes=[list(first.shape), list(second.shape), list(first[:1].shape)],
        max_hidden_error=errors, seconds_runs=samples, median_seconds=medians,
        relative_time=medians["openvino"] / medians["eager"],
        meets_15_percent_feature_speed_and_parity_gate=passing,
        compile_seconds=compile_seconds, no_validation_labels_used=True,
        no_test_scoring=True,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
