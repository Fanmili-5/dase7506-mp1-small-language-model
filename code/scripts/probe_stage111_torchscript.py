"""Compare FP32 TorchScript and eager Stage105 neural features on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.checkpoint) != STAGE105_SHA or args.run_dir.exists():
        raise ValueError("Unexpected checkpoint or existing output directory")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _ = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    features = Features(model.neural).eval()
    ids = (torch.arange(32 * 256).reshape(32, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        traced = torch.jit.trace(features, ids, check_trace=False)
        compiled = torch.jit.optimize_for_inference(torch.jit.freeze(traced))
        full_expected = features(ids)
        full_actual = compiled(ids)
        small_ids = ids[:2]
        small_expected = features(small_ids)
        small_actual = compiled(small_ids)
    errors = {
        "full_max_abs_error": float((full_actual - full_expected).abs().max()),
        "small_max_abs_error": float((small_actual - small_expected).abs().max()),
    }
    if not torch.isfinite(full_actual).all() or not torch.isfinite(small_actual).all():
        raise ValueError("TorchScript returned nonfinite features")
    args.run_dir.mkdir(parents=True)
    script_path = args.run_dir / "neural-features-scripted.pt"
    torch.jit.save(compiled, str(script_path))
    for _ in range(3):
        with torch.inference_mode():
            features(ids)
            compiled(ids)
    times = {"eager": [], "torchscript": []}
    for repeat in range(12):
        order = ("eager", "torchscript") if repeat % 2 == 0 else (
            "torchscript", "eager")
        for name in order:
            started = time.perf_counter()
            with torch.inference_mode():
                (features if name == "eager" else compiled)(ids)
            times[name].append(time.perf_counter() - started)
    result = {
        "purpose": "runtime_feasibility_only_no_validation_or_test_scoring",
        "source_checkpoint_sha256": STAGE105_SHA,
        "torchscript_sha256": sha(script_path),
        "torchscript_bytes": script_path.stat().st_size,
        "shape": list(ids.shape),
        "threads": 4,
        "errors": errors,
        "times_seconds": times,
        "median_eager_seconds": statistics.median(times["eager"]),
        "median_torchscript_seconds": statistics.median(times["torchscript"]),
        "no_validation_or_test_scoring": True,
    }
    (args.run_dir / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
