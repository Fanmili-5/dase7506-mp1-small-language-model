"""Synthetic-only Stage209 exactness, CPU, asset and GPU-memory screen."""
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

from common import PROTOCOL, sha
from train_experiment import training_autocast
from student_stage208_lexical_hierarchy import LexicalHierarchy
import student_stage209_tree_propagation as candidate_source


CONFIG = ROOT / "configs/stage208_lexical_hierarchy_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
ASSET_LIMIT = 64 * 1024**2
REFERENCE_ASSETS = 55_810_412
SOURCE_RESERVE = 524_288


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite Stage209 preflight evidence")
    if CONFIG.read_bytes() != CONTROL.read_bytes():
        raise ValueError("Stage209 changed the matched Stage54 configuration")
    torch.manual_seed(209017)
    torch.set_num_threads(4)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cpu_model = candidate_source.build_model(config).eval()
    old_tree = LexicalHierarchy(2048, 288).eval()
    with torch.no_grad():
        old_tree.node_weight.copy_(cpu_model.lexical.node_weight)
        old_tree.node_bias.copy_(cpu_model.lexical.node_bias)
        old_tree.node_weight.normal_(0, .02)
        old_tree.node_bias.normal_(0, .02)
        cpu_model.lexical.node_weight.copy_(old_tree.node_weight)
        cpu_model.lexical.node_bias.copy_(old_tree.node_bias)
    hidden = torch.randn(2, 12, 288)
    with torch.inference_mode():
        tree_error = float((old_tree(hidden) - cpu_model.lexical(hidden)).abs().max())
        ids = torch.randint(0, 2048, (2, 12), dtype=torch.long)
        logp = cpu_model.predict_log_probs(ids)
        normalization = float(logp.logsumexp(-1).abs().max())
        changed_ids = ids.clone()
        changed_ids[0, -1] = (changed_ids[0, -1] + 1) % 2048
        prefix_error = float((cpu_model.predict_log_probs(changed_ids)[0, :-1]
                              - logp[0, :-1]).abs().max())
        row_error = float((cpu_model.predict_log_probs(ids[:1])[0]
                           - logp[0]).abs().max())
        tree_norm = float(cpu_model.lexical(hidden).logsumexp(-1).abs().max())
        large_hidden = torch.randn(32, 256, 288)
        for _ in range(2):
            cpu_model.lexical(large_hidden)
        timings = []
        for _ in range(8):
            started = time.perf_counter()
            cpu_model.lexical(large_hidden)
            timings.append(time.perf_counter() - started)
    lexical_parameters = sum(parameter.numel() for parameter in
                             tuple(cpu_model.lexical.parameters())
                             + tuple(cpu_model.lexical_gate.parameters()))
    tree_buffer_bytes = sum(buffer.numel() * buffer.element_size()
                            for buffer in cpu_model.lexical.buffers())
    projected_assets = (REFERENCE_ASSETS + 4 * lexical_parameters
                        + tree_buffer_bytes + SOURCE_RESERVE)
    cpu_median = statistics.median(timings)
    del cpu_model, old_tree

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage209 preflight requires BF16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(209017)
    torch.cuda.manual_seed_all(209017)
    model = candidate_source.build_model(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    gpu_ids = torch.randint(0, 2048, (32, 256), device=device)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with training_autocast(device, "bf16"):
        loss, _ = model.rdrop_training_loss(gpu_ids, targets, future)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Stage209 synthetic loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    lexical_grad_norm = float(model.lexical.node_weight.grad.norm())
    optimizer.step()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    gpu_total = torch.cuda.get_device_properties(device).total_memory
    passed = bool(tree_error <= 2e-6 and normalization <= 1e-5
                  and prefix_error <= 1e-5 and row_error <= 1e-5
                  and tree_norm <= 1e-5 and cpu_median <= .50
                  and projected_assets <= ASSET_LIMIT
                  and 0 < grad_norm < float("inf")
                  and 0 < lexical_grad_norm < float("inf")
                  and allocated <= 7 * 1024**3 and reserved < gpu_total)
    result = {
        "status": "synthetic_input_only_preflight", "protocol": PROTOCOL,
        "no_data_split_opened": True, "test_scored": False,
        "config_sha256": sha(CONFIG), "control_config_sha256": sha(CONTROL),
        "model_source_sha256": sha(ROOT / "student_stage209_tree_propagation.py"),
        "source_sha256": sha(Path(__file__)),
        "parent_source_sha256": candidate_source.PARENT_SHA256,
        "tokenizer_sha256": sha(ROOT / "data/tokenizer.json"),
        "max_old_formula_error": tree_error,
        "max_normalization_error": normalization,
        "max_hierarchy_normalization_error": tree_norm,
        "max_future_prefix_error": prefix_error,
        "max_independent_row_error": row_error,
        "lexical_parameters": lexical_parameters,
        "tree_buffer_bytes": tree_buffer_bytes,
        "projected_conservative_assets_bytes": projected_assets,
        "asset_limit_bytes": ASSET_LIMIT,
        "batch": 32, "context": 256, "threads": 4,
        "lexical_head_seconds": timings,
        "median_lexical_head_seconds": cpu_median,
        "synthetic_loss": float(loss.detach()),
        "synthetic_grad_norm": grad_norm,
        "synthetic_lexical_grad_norm": lexical_grad_norm,
        "gpu_peak_allocated_bytes": allocated,
        "gpu_peak_reserved_bytes": reserved,
        "gpu_total_bytes": gpu_total,
        "admit_matched_training_pilot": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage209 fixed feasibility gate failed")


if __name__ == "__main__":
    main()
