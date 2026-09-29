"""Input-only causal, CPU export/timing, and GPU-memory screen for Stage206."""
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

from common import PROTOCOL, sha
from scripts.probe_stage145_attention_allocation import FeatureWrapper, make_portable, ov_model
from train_experiment import training_autocast
import student_stage206_linear_memory_rdrop


REFERENCE = ROOT / "inference_assets/stage143-stage92-features.onnx"
REFERENCE_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
REFERENCE_ASSETS = 55_810_412
CONFIG = ROOT / "configs/stage206_linear_memory_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
ASSET_LIMIT = 64 * 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new Stage206 preflight directory")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Reference feature graph changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control.keys()
               if config.get(key) != control.get(key)}
    if (changed != {"linear_memory_layers", "linear_memory_key_dim"}
            or config["linear_memory_layers"] != [3, 7]
            or config["linear_memory_key_dim"] != 16):
        raise ValueError("Stage206 changed outside the fixed core replacement")

    torch.manual_seed(206017)
    torch.set_num_threads(4)
    ids = torch.randint(0, 2048, (32, 256), dtype=torch.long)
    cpu_model = student_stage206_linear_memory_rdrop.build_model(config).eval()
    with torch.inference_mode():
        small = ids[:2, :32]
        logp = cpu_model.predict_log_probs(small)
        normalization = float(logp.logsumexp(-1).abs().max())
        changed_ids = small.clone()
        changed_ids[0, -1] = (changed_ids[0, -1] + 1) % 2048
        prefix_error = float((cpu_model.predict_log_probs(changed_ids)[0, :-1]
                              - logp[0, :-1]).abs().max())
        row_error = float((cpu_model.predict_log_probs(small[:1])[0]
                           - logp[0]).abs().max())
    eager = FeatureWrapper(cpu_model).eval()
    portable = FeatureWrapper(copy.deepcopy(cpu_model)).eval()
    norm_count = make_portable(portable)
    with torch.inference_mode():
        portable_error = float((portable(ids[:1]) - eager(ids[:1])).abs().max())
    if norm_count == 0 or portable_error > 3e-4:
        raise ValueError("Portable norm parity failed")
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage206-random-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(portable, ids, str(graph), export_params=True,
                          opset_version=17, do_constant_folding=True,
                          input_names=["ids"], output_names=["hidden"],
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
    times = {"reference": [], "candidate": []}
    for _ in range(2):
        reference({"ids": ids.numpy()}); candidate({"ids": ids.numpy()})
    for index in range(8):
        order = (("reference", reference), ("candidate", candidate))
        if index % 2:
            order = tuple(reversed(order))
        for label, model in order:
            started = time.perf_counter()
            model({"ids": ids.numpy()})
            times[label].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in times.items()}
    projected_assets = (graph.stat().st_size + REFERENCE_ASSETS
                        - REFERENCE.stat().st_size + 262_144)
    del cpu_model, eager, portable

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage206 preflight requires BF16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(206017)
    torch.cuda.manual_seed_all(206017)
    gpu_model = student_stage206_linear_memory_rdrop.build_model(config).to(device)
    gpu_model.train()
    optimizer = torch.optim.AdamW(gpu_model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with training_autocast(device, "bf16"):
        loss, _ = gpu_model.rdrop_training_loss(ids.to(device), targets, future)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Stage206 synthetic loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(gpu_model.parameters(), 1.0))
    optimizer.step()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    gpu_total = torch.cuda.get_device_properties(device).total_memory
    passed = bool(normalization <= 1e-5 and prefix_error <= 1e-5
                  and row_error <= 1e-5 and max(errors) <= 3e-4
                  and projected_assets <= ASSET_LIMIT
                  and medians["candidate"] <= 1.25 * medians["reference"]
                  and 0 < grad_norm < float("inf")
                  and allocated <= 7 * 1024**3 and reserved < gpu_total)
    result = {
        "status": "synthetic_input_only_preflight", "protocol": PROTOCOL,
        "no_data_split_opened": True, "test_scored": False,
        "config_sha256": sha(CONFIG), "control_config_sha256": sha(CONTROL),
        "source_sha256": sha(Path(__file__)),
        "model_source_sha256": sha(ROOT / "student_stage206_linear_memory_rdrop.py"),
        "reference_graph_sha256": REFERENCE_SHA,
        "candidate_graph_sha256": sha(graph), "candidate_graph_bytes": graph.stat().st_size,
        "projected_conservative_assets_bytes": projected_assets,
        "asset_limit_bytes": ASSET_LIMIT, "portable_norm_count": norm_count,
        "portable_hidden_max_error": portable_error,
        "openvino_hidden_max_error": max(errors),
        "max_normalization_error": normalization,
        "max_future_prefix_error": prefix_error,
        "max_independent_row_error": row_error,
        "timing_seconds": times, "median_seconds": medians,
        "candidate_to_reference_feature_time": medians["candidate"] / medians["reference"],
        "synthetic_loss": float(loss.detach()), "synthetic_grad_norm": grad_norm,
        "synthetic_optimizer_steps": 1,
        "gpu_peak_allocated_bytes": allocated,
        "gpu_peak_reserved_bytes": reserved, "gpu_total_bytes": gpu_total,
        "admit_matched_training_pilot": passed,
    }
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage206 fixed feasibility gate failed")


if __name__ == "__main__":
    main()
