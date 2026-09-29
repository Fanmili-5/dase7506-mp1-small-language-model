"""Score exact Stage105 with only its frozen feature extractor on OpenVINO CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov
import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from evaluate import score
from train_experiment import atomic_json_dump


STAGE105_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"
GRAPH_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
REFERENCE_BPB = 1.399686163172


class FrozenOpenVinoFeatures:
    def __init__(self, graph: Path):
        core = ov.Core()
        self.compiled = core.compile_model(str(graph), "CPU", {
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
            "INFERENCE_NUM_THREADS": 4,
            "NUM_STREAMS": 1,
            "ENABLE_CPU_PINNING": False,
        })
        precision = self.compiled.get_property("INFERENCE_PRECISION_HINT")
        threads = self.compiled.get_property("INFERENCE_NUM_THREADS")
        streams = self.compiled.get_property("NUM_STREAMS")
        pinning = self.compiled.get_property("ENABLE_CPU_PINNING")
        if precision != ov.Type.f32 or int(threads) != 4 or int(streams) != 1 or pinning:
            raise ValueError("OpenVINO precision/thread settings changed")
        self.properties = dict(precision=str(precision), threads=int(threads),
                               streams=int(streams), cpu_pinning=bool(pinning))

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("OpenVINO feature graph requires CPU input")
        output = self.compiled({"ids": ids.numpy()})
        return torch.from_numpy(next(iter(output.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage142 evidence")
    if sha(args.checkpoint) != STAGE105_SHA or sha(args.graph) != GRAPH_SHA:
        raise ValueError("Unexpected frozen Stage105 checkpoint/feature graph")
    device, precision = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_stage105_gated_singlepass"):
        raise ValueError("Unexpected Stage105 payload")
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    validation = load_data()["validation"]
    source = windows(validation[0], 32)
    inputs = (next(source)[0], next(source)[0])
    with torch.inference_mode():
        eager_outputs = [model.predict_log_probs(ids) for ids in inputs]
    backend = FrozenOpenVinoFeatures(args.graph)
    model.neural.features = backend
    with torch.inference_mode():
        substituted_outputs = [model.predict_log_probs(ids) for ids in inputs]
    max_probability_error = max(
        float((a.exp() - b.exp()).abs().max())
        for a, b in zip(eager_outputs, substituted_outputs))
    max_logp_error = max(
        float((a - b).abs().max())
        for a, b in zip(eager_outputs, substituted_outputs))
    max_log_norm = max(float(b.logsumexp(-1).abs().max())
                       for b in substituted_outputs)
    if (max_probability_error > 3e-6 or max_logp_error > 3e-4
            or max_log_norm > 1e-5):
        raise ValueError("Stage105/OpenVINO full-distribution parity failed")

    substituted = score(model, *validation, device, precision)
    model.neural.__dict__.pop("features")
    eager = score(model, *validation, device, precision)
    if (eager["targets"] != 376599 or substituted["targets"] != 376599
            or abs(eager["bpb"] - REFERENCE_BPB) > 2e-6
            or abs(substituted["bpb"] - eager["bpb"]) > 2e-5):
        raise ValueError("Stage105 complete-validation reference mismatch")
    relative = substituted["seconds"] / eager["seconds"]
    result = dict(
        protocol=PROTOCOL, split="validation", precision="fp32",
        checkpoint_sha256=STAGE105_SHA, graph_sha256=GRAPH_SHA,
        source_sha256=sha(Path(__file__)),
        openvino_version=ov.__version__, compiled_properties=backend.properties,
        max_probability_error=max_probability_error,
        max_logp_error=max_logp_error,
        max_log_normalization_error=max_log_norm,
        eager_bpb=eager["bpb"], openvino_bpb=substituted["bpb"],
        eager_seconds=eager["seconds"], openvino_seconds=substituted["seconds"],
        openvino_relative_time=relative,
        targets=eager["targets"], utf8_bytes=eager["utf8_bytes"],
        passes_full_quality_and_20_percent_speed_gate=(
            substituted["bpb"] < 1.4 and relative <= .8),
        not_asset_qualified_duplicate_backbone=True,
        no_test_scoring=True,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
