"""Input-only FP32 OpenVINO preflight for Stage145 six-attention backbone."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov
import torch
from torch import nn
from tokenizers import Tokenizer

from common import sha, windows
import student_hybrid_conv_rdrop
import student_hybrid_conv_structured


REFERENCE = ROOT / "inference_assets/stage143-stage92-features.onnx"
REFERENCE_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
CONFIG = ROOT / "configs/stage145_six_attention_rdrop.json"


class FeatureWrapper(nn.Module):
    def __init__(self, neural: nn.Module):
        super().__init__()
        self.neural = neural

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.neural.features(ids)


class PortableRMSNorm(nn.Module):
    def __init__(self, original: nn.RMSNorm):
        super().__init__()
        self.eps = original.eps
        self.weight = nn.Parameter(original.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def make_portable(module: nn.Module) -> int:
    count = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, PortableRMSNorm(child))
            count += 1
        else:
            count += make_portable(child)
    return count


def ov_model(core: ov.Core, path: Path):
    compiled = core.compile_model(str(path), "CPU", {
        "INFERENCE_PRECISION_HINT": ov.Type.f32,
        "INFERENCE_NUM_THREADS": 4,
        "NUM_STREAMS": 1,
        "ENABLE_CPU_PINNING": False,
    })
    if (compiled.get_property("INFERENCE_PRECISION_HINT") != ov.Type.f32
            or int(compiled.get_property("INFERENCE_NUM_THREADS")) != 4
            or int(compiled.get_property("NUM_STREAMS")) != 1
            or compiled.get_property("ENABLE_CPU_PINNING")):
        raise ValueError("OpenVINO changed required FP32/four-thread settings")
    return compiled


def bench(model, ids, repeats: int = 8) -> list[float]:
    for _ in range(2):
        model({"ids": ids.numpy()})
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model({"ids": ids.numpy()})
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Stage143 reference feature graph changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["conv_layers"] != [4, 8]:
        raise ValueError("Unexpected Stage145 token-mixing allocation")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    neural_config = student_hybrid_conv_rdrop.inference_config(config)
    neural = student_hybrid_conv_structured.build_model(neural_config).eval()
    eager = FeatureWrapper(neural).eval()
    portable = FeatureWrapper(copy.deepcopy(neural)).eval()
    replaced = make_portable(portable)
    if replaced == 0:
        raise ValueError("No RMSNorm was made portable")
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    val_path = ROOT / "data/wikitext_validation.txt"
    tok_path = ROOT / "data/tokenizer.json"
    if (sha(val_path) != manifest["sha256"][val_path.name]
            or sha(tok_path) != manifest["sha256"][tok_path.name]):
        raise ValueError("Fixed validation inputs changed")
    tokenizer = Tokenizer.from_file(str(tok_path))
    tokens = torch.tensor(tokenizer.encode(val_path.read_text(encoding="utf-8")).ids)
    ids, _ = next(windows(tokens, batch_size=32))
    with torch.inference_mode():
        portable_error = float((portable(ids) - eager(ids)).abs().max())
    if portable_error > 3e-4:
        raise ValueError(f"Portable norm mismatch: {portable_error}")
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage145-six-attention-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            portable, ids, str(graph), export_params=True, opset_version=17,
            do_constant_folding=True, input_names=["ids"], output_names=["hidden"],
            dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}})
    core = ov.Core()
    reference = ov_model(core, REFERENCE)
    candidate = ov_model(core, graph)
    with torch.inference_mode():
        expected = eager(ids)
        actual = torch.from_numpy(next(iter(candidate({"ids": ids.numpy()}).values())))
        hidden_error = float((actual - expected).abs().max())
        one = ids[:1]
        one_expected = eager(one)
        one_actual = torch.from_numpy(next(iter(candidate({"ids": one.numpy()}).values())))
        one_error = float((one_actual - one_expected).abs().max())
    # Alternate to reduce host drift; both graphs use identical FP32 settings.
    timing = {"reference": [], "candidate": []}
    for _ in range(2):
        reference({"ids": ids.numpy()}); candidate({"ids": ids.numpy()})
    for i in range(8):
        for label, model in (("reference", reference), ("candidate", candidate)) if i % 2 == 0 else (("candidate", candidate), ("reference", reference)):
            start = time.perf_counter()
            model({"ids": ids.numpy()})
            timing[label].append(time.perf_counter() - start)
    medians = {label: statistics.median(samples) for label, samples in timing.items()}
    result = {
        "purpose": "input_only_random_weight_resource_screen",
        "split": "validation_inputs_no_labels",
        "test_scored": False,
        "config_sha256": sha(CONFIG),
        "source_sha256": sha(Path(__file__)),
        "reference_graph_sha256": REFERENCE_SHA,
        "candidate_graph_sha256": sha(graph),
        "candidate_graph_bytes": graph.stat().st_size,
        "neural_parameters": sum(p.numel() for p in neural.parameters()),
        "portable_norm_count": replaced,
        "max_portable_hidden_error": portable_error,
        "max_openvino_hidden_error": max(hidden_error, one_error),
        "timing_seconds": timing,
        "median_seconds": medians,
        "candidate_to_reference_feature_time": medians["candidate"] / medians["reference"],
        "eligible_for_pilot": bool(max(hidden_error, one_error) <= 3e-4
                                   and graph.stat().st_size <= 42 * 1024 * 1024
                                   and medians["candidate"] <= 1.25 * medians["reference"]),
    }
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
