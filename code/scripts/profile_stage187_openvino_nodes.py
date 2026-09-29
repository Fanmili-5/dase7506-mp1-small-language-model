"""Profile unchanged Stage143 OpenVINO graph nodes on validation inputs only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import openvino as ov
import torch
from tokenizers import Tokenizer

from common import PROTOCOL, setup, sha, windows


GRAPH = ROOT / "inference_assets/stage143-stage92-features.onnx"
CHECKPOINT = ROOT / "checkpoints/stage143-openvino-order6.pt"
GRAPH_SHA256 = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
CHECKPOINT_SHA256 = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
SOURCE_SHA256 = "1c31716eccd3bffdd2664b9ba654950a5d64d3ca0ad4e4ded817b1acbf6c9615"


def validation_inputs(batches: int) -> list[np.ndarray]:
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("tokenizer.json", "wikitext_validation.txt"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError(f"Changed fixed data file: {name}")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    text = (ROOT / "data/wikitext_validation.txt").read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    inputs = [x.numpy() for x, _ in itertools.islice(
        windows(ids, batch_size=32), batches)]
    if len(inputs) != batches or any(x.shape != (32, 256) for x in inputs):
        raise ValueError("Expected full independent validation input batches")
    return inputs


def checked_compile(core: ov.Core, profiling: bool):
    options = {
        "INFERENCE_PRECISION_HINT": ov.Type.f32,
        "INFERENCE_NUM_THREADS": 1,
        "NUM_STREAMS": 1,
        "ENABLE_CPU_PINNING": False,
        ov.properties.enable_profiling: profiling,
    }
    compiled = core.compile_model(str(GRAPH), "CPU", options)
    if (compiled.get_property("INFERENCE_PRECISION_HINT") != ov.Type.f32
            or int(compiled.get_property("INFERENCE_NUM_THREADS")) != 1
            or int(compiled.get_property("NUM_STREAMS")) != 1
            or compiled.get_property("ENABLE_CPU_PINNING")):
        raise ValueError("OpenVINO precision/thread/stream contract changed")
    return compiled.create_infer_request()


def first_output(request, ids: np.ndarray) -> np.ndarray:
    output = request.infer({"ids": ids})
    if len(output) != 1:
        raise ValueError("Unexpected graph output count")
    value = next(iter(output.values()))
    if value.shape != (*ids.shape, 288) or value.dtype != np.float32:
        raise ValueError("Unexpected graph output shape or dtype")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite graph output")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batches < 1 or args.output.exists():
        parser.error("Use at least one batch and a new output file")
    if sha(GRAPH) != GRAPH_SHA256 or sha(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Frozen Stage143 graph/checkpoint changed")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("implementation") != "student_stage143_openvino_singlepass"
            or sha(ROOT / "student_stage143_openvino_singlepass.py") != SOURCE_SHA256):
        raise ValueError("Frozen Stage143 source/contract changed")
    setup("cpu", "fp32", 1)
    inputs = validation_inputs(args.batches)
    core = ov.Core()
    reference = checked_compile(core, False)
    profiled = checked_compile(core, True)
    ref_first = first_output(reference, inputs[0]).copy()
    prof_first = first_output(profiled, inputs[0]).copy()
    max_diff = float(np.max(np.abs(ref_first - prof_first)))
    if max_diff > 1e-3:
        raise ValueError(f"Profiled/unprofiled output mismatch: {max_diff}")

    node_totals = defaultdict(lambda: [0.0, 0.0, 0])
    type_totals = defaultdict(lambda: [0.0, 0.0, 0])
    wall = {"reference": [], "profiled": []}
    node_seconds_by_batch = []
    requests = {"reference": reference, "profiled": profiled}
    for index, ids in enumerate(inputs):
        order = ("reference", "profiled") if index % 2 == 0 else ("profiled", "reference")
        for name in order:
            started = time.perf_counter()
            first_output(requests[name], ids)
            wall[name].append(time.perf_counter() - started)
            if name != "profiled":
                continue
            node_seconds = 0.0
            for info in profiled.profiling_info:
                real = info.real_time.total_seconds()
                cpu = info.cpu_time.total_seconds()
                key = (info.node_name, info.node_type, info.exec_type)
                for target, label in ((node_totals, key), (type_totals, info.node_type)):
                    target[label][0] += real
                    target[label][1] += cpu
                    target[label][2] += 1
                node_seconds += real
            node_seconds_by_batch.append(node_seconds)
    if not node_totals or max(node_seconds_by_batch, default=0) <= 0:
        raise ValueError("OpenVINO returned no usable node timing")

    def rows(source, by_node: bool):
        ordered = sorted(source.items(), key=lambda pair: pair[1][0], reverse=True)
        return [dict(name=key[0], type=key[1], exec_type=key[2],
                     real_seconds=values[0], cpu_seconds=values[1], calls=values[2])
                if by_node else dict(type=key, real_seconds=values[0],
                                     cpu_seconds=values[1], calls=values[2])
                for key, values in ordered]

    result = dict(
        protocol=PROTOCOL, purpose="stage187_linux_openvino_node_profile",
        split="validation_inputs_only", scored_targets=0, no_test_scoring=True,
        precision="fp32", threads=1, platform=platform.platform(),
        openvino_version=ov.__version__, checkpoint_sha256=sha(CHECKPOINT),
        graph_sha256=sha(GRAPH), source_sha256=SOURCE_SHA256,
        script_sha256=sha(Path(__file__)), batches=args.batches,
        input_rows=args.batches * 32, max_profiled_reference_hidden_diff=max_diff,
        wall_seconds_by_batch=wall,
        wall_seconds_total={name: sum(values) for name, values in wall.items()},
        profiled_node_real_seconds_by_batch=node_seconds_by_batch,
        top_nodes=rows(node_totals, True)[:30], node_types=rows(type_totals, False),
        caution="Profiled node times may overlap or differ from unprofiled wall time; this is not a resource gate.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "max_profiled_reference_hidden_diff", "wall_seconds_total", "node_types", "top_nodes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
