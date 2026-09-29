"""Random-weight input-only resource preflight for a tiny second expert."""
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
import student_hybrid_conv_rdrop
import student_hybrid_conv_structured

CONFIG = ROOT / "configs/stage159_compact_complement_rdrop.json"
STAGE143_ASSETS = 55_810_412
HEAD_SOURCE_RESERVE = 1_500_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new resource preflight directory")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Stage143 reference graph changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["width"] != 128 or config["depth"] != 4
            or config["conv_layers"] != [2, 4]
            or config["deep_supervision_layers"] != [2, 3]):
        raise ValueError("Unexpected compact complement architecture")
    torch.manual_seed(17)
    torch.set_num_threads(4)
    neural = student_hybrid_conv_structured.build_model(
        student_hybrid_conv_rdrop.inference_config(config)).eval()
    eager = FeatureWrapper(neural).eval()
    portable = FeatureWrapper(copy.deepcopy(neural)).eval()
    rewritten_norms = make_portable(portable)
    if rewritten_norms == 0:
        raise ValueError("RMSNorm rewrite did not occur")
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError(f"Fixed validation input changed: {name}")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    tokens = torch.tensor(tokenizer.encode(
        (ROOT / "data/wikitext_validation.txt").read_text(encoding="utf-8")).ids)
    ids, _ = next(windows(tokens, batch_size=32))
    with torch.inference_mode():
        portable_error = float((portable(ids) - eager(ids)).abs().max())
    if portable_error > 3e-4:
        raise ValueError(f"Portable norm mismatch: {portable_error}")
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage159-random-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(portable, ids, str(graph), export_params=True,
                          opset_version=17, do_constant_folding=True,
                          input_names=["ids"], output_names=["hidden"],
                          dynamic_axes={"ids": {0: "batch"},
                                        "hidden": {0: "batch"}})
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
        for label, model in pairs:
            before = time.perf_counter()
            model({"ids": ids.numpy()})
            timing[label].append(time.perf_counter() - before)
    median = {label: statistics.median(values) for label, values in timing.items()}
    ratio = median["candidate"] / median["reference"]
    projected_assets = STAGE143_ASSETS + graph.stat().st_size + HEAD_SOURCE_RESERVE
    result = dict(
        purpose="random_weight_input_only_compact_complement_resource_screen",
        split="validation_inputs_no_labels", no_test_scoring=True,
        config_sha256=sha(CONFIG), source_sha256=sha(Path(__file__)),
        reference_graph_sha256=REFERENCE_SHA,
        candidate_graph_sha256=sha(graph),
        candidate_graph_bytes=graph.stat().st_size,
        neural_parameters=sum(p.numel() for p in neural.parameters()),
        portable_norm_count=rewritten_norms,
        max_portable_hidden_error=portable_error,
        max_openvino_hidden_error=max(errors),
        timing_seconds=timing, median_seconds=median,
        candidate_to_reference_feature_time=ratio,
        stage143_asset_bytes=STAGE143_ASSETS,
        compact_head_source_reserve_bytes=HEAD_SOURCE_RESERVE,
        conservative_projected_combined_assets_bytes=projected_assets,
        pass_hidden_parity=max(errors) <= 3e-4,
        pass_input_only_speed=ratio <= .35,
        pass_projected_assets=projected_assets <= 64 * 1024 ** 2,
    )
    result["training_pilot_authorized"] = all(result[key] for key in (
        "pass_hidden_parity", "pass_input_only_speed", "pass_projected_assets"))
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
