"""Isolated FP32 ONNX Runtime feasibility probe for Stage105 neural features.

This does not create an inference candidate or score validation labels. It
compares feature tensors and repeated single-batch wall time only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from common import make_model, setup, sha


STAGE105_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"


class Features(nn.Module):
    def __init__(self, neural: nn.Module):
        super().__init__()
        self.neural = neural

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.neural.features(ids)


class ManualRMSNorm(nn.Module):
    """Export-only spelling of the same FP32 RMS normalization."""

    def __init__(self, original: nn.RMSNorm):
        super().__init__()
        self.eps = float(original.eps)
        self.register_buffer("weight", original.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def replace_rms_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, ManualRMSNorm(child))
        else:
            replace_rms_norms(child)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new probe directory")
    if sha(args.checkpoint) != STAGE105_SHA:
        raise ValueError("Unexpected source checkpoint")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    ids = (torch.arange(32 * 256).reshape(32, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        original_hidden = model.neural.features(ids).numpy()
    features = Features(model.neural).eval()
    replace_rms_norms(features)
    args.run_dir.mkdir(parents=True)
    onnx_path = args.run_dir / "stage105-neural-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            features, (ids,), str(onnx_path), input_names=["ids"],
            output_names=["hidden"], opset_version=18, dynamo=False,
            dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}},
            do_constant_folding=True,
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = features(ids).numpy()
    actual = session.run(["hidden"], {"ids": ids.numpy()})[0]
    max_abs_error = float(np.max(np.abs(actual - original_hidden)))
    rms_error = float(np.sqrt(np.mean((actual - original_hidden) ** 2)))
    manual_error = float(np.max(np.abs(expected - original_hidden)))
    if not np.isfinite(actual).all():
        raise ValueError("ONNX features are non-finite")
    for _ in range(3):
        with torch.inference_mode():
            features(ids)
        session.run(["hidden"], {"ids": ids.numpy()})
    timings = {"pytorch": [], "onnxruntime": []}
    for repeat in range(12):
        order = ("pytorch", "onnxruntime") if repeat % 2 == 0 else (
            "onnxruntime", "pytorch")
        for backend in order:
            started = time.perf_counter()
            if backend == "pytorch":
                with torch.inference_mode():
                    features(ids)
            else:
                session.run(["hidden"], {"ids": ids.numpy()})
            timings[backend].append(time.perf_counter() - started)
    result = {
        "purpose": "runtime_feasibility_only_no_validation_labels",
        "source_checkpoint_sha256": STAGE105_SHA,
        "onnx_sha256": sha(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "shape": list(ids.shape),
        "threads": 4,
        "max_feature_absolute_error": max_abs_error,
        "rms_feature_error": rms_error,
        "manual_rms_norm_absolute_error": manual_error,
        "times_seconds": timings,
        "median_pytorch_seconds": statistics.median(timings["pytorch"]),
        "median_onnxruntime_seconds": statistics.median(timings["onnxruntime"]),
        "no_validation_or_test_scoring": True,
    }
    (args.run_dir / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
