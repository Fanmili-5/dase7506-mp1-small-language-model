"""Input-only Windows OpenVINO feasibility screen for Stage186."""
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
from tokenizers import Tokenizer

from common import sha, windows
from scripts.probe_stage145_attention_allocation import (
    FeatureWrapper, make_portable, ov_model, REFERENCE, REFERENCE_SHA,
)
import student_stage186_narrow_parallel_rdrop as training_module
import student_stage186_narrow_parallel_structured as inference_module


CONFIG = ROOT / "configs/stage186_narrow_parallel_rdrop.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Stage143 reference feature graph changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["width"] != 288 or config["depth"] != 8
            or config["conv_layers"] != [2, 4, 6, 8]
            or config["parallel_heads"] != 2 or config["parallel_head_dim"] != 36):
        raise ValueError("Unexpected Stage186 configuration")
    torch.manual_seed(17)
    torch.set_num_threads(4)
    neural = inference_module.build_model(
        training_module.inference_config(config)).eval()
    eager = FeatureWrapper(neural).eval()
    portable = FeatureWrapper(copy.deepcopy(neural)).eval()
    replaced = make_portable(portable)
    if replaced == 0:
        raise ValueError("No RMSNorm rewritten for ONNX export")
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    val_path = ROOT / "data/wikitext_validation.txt"
    tok_path = ROOT / "data/tokenizer.json"
    if (sha(val_path) != manifest["sha256"][val_path.name]
            or sha(tok_path) != manifest["sha256"][tok_path.name]):
        raise ValueError("Supplied validation/tokenizer files changed")
    tokenizer = Tokenizer.from_file(str(tok_path))
    tokens = torch.tensor(tokenizer.encode(val_path.read_text(encoding="utf-8")).ids)
    ids, _ = next(windows(tokens, batch_size=32))
    with torch.inference_mode():
        norm_error = float((portable(ids) - eager(ids)).abs().max())
    if norm_error > 3e-4:
        raise ValueError("Portable RMSNorm rewrite changed the model")

    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage186-narrow-parallel-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            portable, ids, str(graph), export_params=True, opset_version=17,
            do_constant_folding=True, input_names=["ids"], output_names=["hidden"],
            dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}})
    core = ov.Core()
    reference = ov_model(core, REFERENCE)
    candidate = ov_model(core, graph)
    errors = []
    with torch.inference_mode():
        for batch in (ids, ids[:1]):
            expected = eager(batch)
            actual = torch.from_numpy(next(iter(candidate({"ids": batch.numpy()}).values())))
            errors.append(float((actual - expected).abs().max()))
    timing = {"reference": [], "candidate": []}
    for _ in range(2):
        reference({"ids": ids.numpy()}); candidate({"ids": ids.numpy()})
    for index in range(8):
        pairs = (("reference", reference), ("candidate", candidate))
        if index % 2:
            pairs = tuple(reversed(pairs))
        for label, compiled in pairs:
            started = time.perf_counter()
            compiled({"ids": ids.numpy()})
            timing[label].append(time.perf_counter() - started)
    median = {label: statistics.median(values) for label, values in timing.items()}
    projected_assets = graph.stat().st_size + 25_000_000 + 200_000
    result = dict(
        purpose="stage186_input_only_random_weight_resource_screen",
        split="validation_inputs_no_labels", test_scored=False,
        config_sha256=sha(CONFIG), source_sha256=sha(Path(__file__)),
        parallel_module_sha256=sha(ROOT / "student_stage186_narrow_parallel.py"),
        training_module_sha256=sha(ROOT / "student_stage186_narrow_parallel_rdrop.py"),
        inference_module_sha256=sha(ROOT / "student_stage186_narrow_parallel_structured.py"),
        stage145_utilities_sha256=sha(ROOT / "scripts/probe_stage145_attention_allocation.py"),
        reference_graph_sha256=REFERENCE_SHA, candidate_graph_sha256=sha(graph),
        candidate_graph_bytes=graph.stat().st_size,
        neural_parameters=sum(parameter.numel() for parameter in neural.parameters()),
        portable_norm_count=replaced, max_portable_hidden_error=norm_error,
        max_openvino_hidden_error=max(errors), timing_seconds=timing,
        median_seconds=median,
        candidate_to_reference_feature_time=median["candidate"] / median["reference"],
        conservative_projected_assets_bytes=projected_assets,
        eligible_for_pilot=bool(max(errors) <= 3e-4
                                and projected_assets <= 64 * 1024 * 1024
                                and median["candidate"] <= 1.25 * median["reference"]),
        warning="Input-only random-weight feasibility, not full-predictor score or resource qualification",
    )
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
