"""Synthetic-only FP32 OpenVINO screen of the exact Stage209 lexical head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov
import torch

from common import PROTOCOL, sha
import student_stage209_tree_propagation as model_source


STAGE209_PREFLIGHT = ROOT / "results/stage209-preflight-a.json"
REFERENCE_ASSETS = 55_810_412
ASSET_LIMIT = 64 * 1024**2
SOURCE_RESERVE = 524_288


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Refusing to overwrite Stage210 preflight evidence")
    previous = json.loads(STAGE209_PREFLIGHT.read_text(encoding="utf-8-sig"))
    if (previous.get("status") != "synthetic_input_only_preflight"
            or previous.get("admit_matched_training_pilot") is not False
            or previous.get("no_data_split_opened") is not True
            or previous.get("model_source_sha256") != sha(ROOT / "student_stage209_tree_propagation.py")
            or previous.get("max_old_formula_error", 1) > 2e-6
            or previous.get("gpu_peak_allocated_bytes", 10**20) > 7 * 1024**3
            or previous.get("projected_conservative_assets_bytes", 10**20) > ASSET_LIMIT):
        raise ValueError("Stage209 parent feasibility record mismatch")
    torch.manual_seed(210017)
    torch.set_num_threads(4)
    head = model_source.PropagatedLexicalHierarchy(2048, 288).eval()
    with torch.no_grad():
        head.node_weight.normal_(0, .02)
        head.node_bias.normal_(0, .02)
    hidden = torch.randn(32, 256, 288)
    with torch.inference_mode():
        eager_large = head(hidden)
        eager_small = head(hidden[:1])
        norm_error = float(eager_large.logsumexp(-1).abs().max())
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage210-random-lexical-head.onnx"
    with torch.inference_mode():
        torch.onnx.export(head, hidden, str(graph), opset_version=17,
                          export_params=True, do_constant_folding=True,
                          input_names=["hidden"], output_names=["logp"],
                          dynamic_axes={"hidden": {0: "batch"},
                                        "logp": {0: "batch"}})
    core = ov.Core()
    compiled = core.compile_model(str(graph), "CPU", {
        "INFERENCE_PRECISION_HINT": ov.Type.f32,
        "INFERENCE_NUM_THREADS": 4,
        "NUM_STREAMS": 1,
        "ENABLE_CPU_PINNING": False,
    })
    if (compiled.get_property("INFERENCE_PRECISION_HINT") != ov.Type.f32
            or int(compiled.get_property("INFERENCE_NUM_THREADS")) != 4
            or int(compiled.get_property("NUM_STREAMS")) != 1
            or compiled.get_property("ENABLE_CPU_PINNING")):
        raise ValueError("Stage210 CPU FP32 runtime settings changed")
    errors = []
    for sample, eager in ((hidden[:1], eager_small), (hidden, eager_large)):
        actual = torch.from_numpy(next(iter(compiled({"hidden": sample.numpy()}).values())))
        errors.append(float((actual - eager).abs().max()))
        norm_error = max(norm_error, float(actual.logsumexp(-1).abs().max()))
    for _ in range(2):
        compiled({"hidden": hidden.numpy()})
    timings = []
    for _ in range(8):
        started = time.perf_counter()
        compiled({"hidden": hidden.numpy()})
        timings.append(time.perf_counter() - started)
    median = statistics.median(timings)
    projected_assets = REFERENCE_ASSETS + graph.stat().st_size + SOURCE_RESERVE
    passed = bool(max(errors) <= 3e-4 and norm_error <= 1e-5
                  and median <= .40 and projected_assets <= ASSET_LIMIT)
    result = {
        "status": "synthetic_input_only_backend_preflight",
        "protocol": PROTOCOL, "no_data_split_opened": True,
        "test_scored": False, "precision": "fp32", "threads": 4,
        "source_sha256": sha(Path(__file__)),
        "model_source_sha256": sha(ROOT / "student_stage209_tree_propagation.py"),
        "stage209_preflight_sha256": sha(STAGE209_PREFLIGHT),
        "graph_sha256": sha(graph), "graph_bytes": graph.stat().st_size,
        "max_openvino_eager_logp_error": max(errors),
        "max_normalization_error": norm_error,
        "head_seconds": timings, "median_head_seconds": median,
        "projected_conservative_assets_bytes": projected_assets,
        "asset_limit_bytes": ASSET_LIMIT,
        "admit_matched_training_pilot": passed,
    }
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage210 fixed backend feasibility gate failed")


if __name__ == "__main__":
    main()
