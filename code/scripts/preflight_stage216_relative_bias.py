"""Stage216 input-only causal, OpenVINO CPU/asset, and CUDA-memory screen."""
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

from common import PROTOCOL, sha
from scripts.probe_stage145_attention_allocation import FeatureWrapper, make_portable, ov_model
from train_experiment import training_autocast
import student_hybrid_conv_rdrop
import student_stage216_relative_bias


REFERENCE = ROOT / "inference_assets/stage143-stage92-features.onnx"
REFERENCE_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
REFERENCE_ASSETS = 55_810_412
CONFIG = ROOT / "configs/stage216_relative_bias_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
ASSET_LIMIT = 64 * 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new Stage216 preflight directory")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Reference feature graph changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control_config = json.loads(CONTROL.read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control_config.keys()
               if config.get(key) != control_config.get(key)}
    if changed != {"relative_bias_buckets"} or config["relative_bias_buckets"] != 32:
        raise ValueError("Stage216 must differ only in its fixed distance bias")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    control = student_hybrid_conv_rdrop.build_model(control_config).eval()
    torch.manual_seed(17)
    candidate = student_stage216_relative_bias.build_model(config).eval()
    for name, expected in control.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], expected, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(216017)
    ids = torch.randint(0, 2048, (32, 256), generator=generator)
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    train_prefix = tokenizer.encode(
        (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")[:1024]
    ).ids[:64]
    with torch.inference_mode():
        random_control = control.predict_log_probs(ids[:2, :64])
        random_candidate = candidate.predict_log_probs(ids[:2, :64])
        train_ids = torch.tensor(train_prefix, dtype=torch.long)[None]
        train_error = float((control.predict_log_probs(train_ids)
                             - candidate.predict_log_probs(train_ids)).abs().max())
        zero_error = max(float((random_control - random_candidate).abs().max()), train_error)
        normalization = float(random_candidate.logsumexp(-1).abs().max())
        modified = ids[:2, :64].clone()
        modified[:, 32:] = torch.randint(0, 2048, (2, 32), generator=generator)
        prefix_error = float((candidate.predict_log_probs(modified)[:, :32]
                              - random_candidate[:, :32]).abs().max())
        row_error = float((candidate.predict_log_probs(ids[:1, :64])[0]
                           - random_candidate[0]).abs().max())
    eager = FeatureWrapper(candidate).eval()
    portable = FeatureWrapper(copy.deepcopy(candidate)).eval()
    portable_norms = make_portable(portable)
    with torch.inference_mode():
        portable_error = float((portable(ids[:1]) - eager(ids[:1])).abs().max())
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage216-random-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            portable, ids, str(graph), export_params=True, opset_version=17,
            do_constant_folding=True, input_names=["ids"], output_names=["hidden"],
            dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}},
        )
    core = ov.Core()
    reference = ov_model(core, REFERENCE)
    compiled = ov_model(core, graph)
    ov_errors = []
    with torch.inference_mode():
        for batch in (ids, ids[:1]):
            actual = torch.from_numpy(next(iter(compiled({"ids": batch.numpy()}).values())))
            ov_errors.append(float((actual - eager(batch)).abs().max()))
    times = {"reference": [], "candidate": []}
    for _ in range(2):
        reference({"ids": ids.numpy()})
        compiled({"ids": ids.numpy()})
    for index in range(8):
        order = (("reference", reference), ("candidate", compiled))
        if index % 2:
            order = tuple(reversed(order))
        for label, model in order:
            started = time.perf_counter()
            model({"ids": ids.numpy()})
            times[label].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in times.items()}
    projected_assets = (REFERENCE_ASSETS - REFERENCE.stat().st_size
                        + graph.stat().st_size + 262_144)
    del control, candidate, eager, portable
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage216 preflight requires BF16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    gpu = student_stage216_relative_bias.build_model(config).to(device).train()
    optimizer = torch.optim.AdamW(gpu.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=.1)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    losses, grad_norms = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, "bf16"):
            loss, _ = gpu.rdrop_training_loss(ids.to(device), targets, future)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Stage216 synthetic loss")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(gpu.parameters(), 1.0))
        if not 0 < grad_norm < float("inf"):
            raise FloatingPointError("Nonfinite Stage216 synthetic gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        grad_norms.append(grad_norm)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    gpu_total = torch.cuda.get_device_properties(device).total_memory
    feature_ratio = medians["candidate"] / medians["reference"]
    passed = bool(zero_error <= 1e-5 and normalization <= 1e-5
                  and prefix_error <= 3e-5 and row_error <= 1e-5
                  and portable_norms > 0 and portable_error <= 3e-4
                  and max(ov_errors) <= 3e-4 and feature_ratio <= 1.30
                  and projected_assets <= ASSET_LIMIT and reserved < 8 * 1024**3
                  and reserved < gpu_total)
    result = {
        "status": "synthetic_input_only_preflight", "protocol": PROTOCOL,
        "no_validation_or_test_scoring": True,
        "config_sha256": sha(CONFIG), "control_config_sha256": sha(CONTROL),
        "source_sha256": sha(Path(__file__)),
        "model_source_sha256": sha(ROOT / "student_stage216_relative_bias.py"),
        "reference_graph_sha256": REFERENCE_SHA,
        "candidate_graph_sha256": sha(graph), "candidate_graph_bytes": graph.stat().st_size,
        "projected_conservative_assets_bytes": projected_assets,
        "asset_limit_bytes": ASSET_LIMIT, "zero_start_max_logp_error": zero_error,
        "max_normalization_error": normalization,
        "max_future_prefix_error": prefix_error,
        "max_independent_row_error": row_error,
        "portable_norm_count": portable_norms,
        "portable_hidden_max_error": portable_error,
        "openvino_hidden_max_error": max(ov_errors),
        "timing_seconds": times, "median_seconds": medians,
        "candidate_to_reference_feature_time": feature_ratio,
        "synthetic_losses": losses, "synthetic_grad_norms": grad_norms,
        "synthetic_optimizer_steps": 2,
        "gpu_peak_allocated_bytes": allocated,
        "gpu_peak_reserved_bytes": reserved,
        "gpu_total_bytes": gpu_total,
        "admit_matched_training_pilot": passed,
    }
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage216 fixed input-only feasibility gate failed")


if __name__ == "__main__":
    main()
