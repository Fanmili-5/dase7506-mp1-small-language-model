"""Locate the input-only Stage204 ONNX/OpenVINO disagreement; no data splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import openvino as ov
import torch
from torch import nn

from common import sha
from scripts.probe_stage145_attention_allocation import FeatureWrapper
import student_stage204_hashed_bigram_rdrop


class PairIds(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.bigram_ids(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite export diagnosis")
    config = json.loads((ROOT / "configs/stage204_hashed_bigram_rdrop.json").read_text())
    torch.manual_seed(204017)
    torch.set_num_threads(4)
    ids = torch.randint(0, 2048, (32, 256), dtype=torch.long)
    model = student_stage204_hashed_bigram_rdrop.build_model(config).eval()
    wrapper = FeatureWrapper(model).eval()
    with torch.inference_mode():
        expected = wrapper(ids).numpy()
    session = ort.InferenceSession(str(args.graph), providers=["CPUExecutionProvider"])
    ort_hidden = session.run(None, {"ids": ids.numpy()})[0]
    core = ov.Core()
    compiled = core.compile_model(str(args.graph), "CPU", {
        "INFERENCE_PRECISION_HINT": ov.Type.f32, "INFERENCE_NUM_THREADS": 4,
        "NUM_STREAMS": 1, "ENABLE_CPU_PINNING": False,
    })
    ov_hidden = next(iter(compiled({"ids": ids.numpy()}).values()))
    pair_graph = args.output.with_suffix(".pairs.onnx")
    pair_wrapper = PairIds(model).eval()
    with torch.inference_mode():
        torch.onnx.export(pair_wrapper, ids, str(pair_graph),
                          export_params=True, opset_version=17,
                          input_names=["ids"], output_names=["pair_ids"],
                          dynamic_axes={"ids": {0: "batch"},
                                        "pair_ids": {0: "batch"}})
        expected_pairs = model.bigram_ids(ids).numpy()
    ort_pairs = ort.InferenceSession(
        str(pair_graph), providers=["CPUExecutionProvider"]
    ).run(None, {"ids": ids.numpy()})[0]
    ov_pair_model = core.compile_model(str(pair_graph), "CPU")
    ov_pairs = next(iter(ov_pair_model({"ids": ids.numpy()}).values()))
    result = {
        "purpose": "stage204_input_only_export_disagreement",
        "test_scored": False, "no_data_split_opened": True,
        "feature_graph_sha256": sha(args.graph),
        "pair_graph_sha256": sha(pair_graph),
        "eager_vs_ort_feature_max_abs": float(np.max(np.abs(expected - ort_hidden))),
        "eager_vs_openvino_feature_max_abs": float(np.max(np.abs(expected - ov_hidden))),
        "ort_vs_openvino_feature_max_abs": float(np.max(np.abs(ort_hidden - ov_hidden))),
        "eager_vs_ort_pair_mismatches": int(np.count_nonzero(expected_pairs != ort_pairs)),
        "eager_vs_openvino_pair_mismatches": int(np.count_nonzero(expected_pairs != ov_pairs)),
        "pair_count": int(expected_pairs.size),
        "eager_pair_samples": expected_pairs[0, :12].tolist(),
        "ort_pair_samples": ort_pairs[0, :12].tolist(),
        "openvino_pair_samples": ov_pairs[0, :12].tolist(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
