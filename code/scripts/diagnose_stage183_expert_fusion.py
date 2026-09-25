"""Validation-only normalized arithmetic/geometric fusion screen for Stage143."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows


CHECKPOINT_SHA256 = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
GRAPH_SHA256 = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
REFERENCE_BPB = 1.399686162042141
EXPECTED_TARGETS = 376_599
EXPECTED_BYTES = 1_148_007
EXPECTED_WINDOWS = 1_472
BETAS = (0.0, 0.1, 0.25, 0.5, 1.0)


def geometric_expert(neural: torch.Tensor, count: torch.Tensor,
                     weight: torch.Tensor) -> torch.Tensor:
    if (neural.shape != count.shape or neural.shape[:-1] != weight.shape
            or neural.shape[-1] != 2048 or not torch.isfinite(neural).all()
            or not torch.isfinite(count).all() or not torch.isfinite(weight).all()
            or (neural < 0).any() or (count < 0).any()
            or (weight <= 0).any() or (weight >= 1).any()):
        raise ValueError("Invalid frozen expert distributions or weights")
    logits = ((1 - weight).unsqueeze(-1) * neural.clamp_min(1e-30).log()
              + weight.unsqueeze(-1) * count.clamp_min(1e-12).log())
    return logits.softmax(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("Use a new output path and a positive thread count")
    if sha(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Stage143 checkpoint changed")
    graph = ROOT / "inference_assets/stage143-stage92-features.onnx"
    if sha(graph) != GRAPH_SHA256:
        raise ValueError("Stage143 feature graph changed")
    device, precision = setup("cpu", "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("implementation") != "student_stage143_openvino_singlepass"):
        raise ValueError("Unexpected Stage143 checkpoint")
    model, implementation_sha = make_model(checkpoint["implementation"],
                                            checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    validation, byte_count = load_data()["validation"]
    if byte_count != EXPECTED_BYTES:
        raise ValueError("Validation byte coverage changed")

    original_add = model.ngram.add_collected
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_add(result, weight, terms, suffix):
        if captured:
            raise ValueError("Expected one count addition per batch")
        neural = result.detach().clone() / (1 - weight).unsqueeze(-1)
        count = suffix[:, None] * model.ngram.unigram[None, :]
        for rows, columns, mass in reversed(terms):
            count[rows, columns] = count[rows, columns] + mass
        count = count.view_as(result)
        captured.append((neural, count, weight.detach().clone()))
        return original_add(result, weight, terms, suffix)

    model.ngram.add_collected = capture_add
    half_nll = [{beta: 0.0 for beta in BETAS} for _ in range(2)]
    half_targets = [0, 0]
    max_reconstruction = 0.0
    max_expert_norm_error = 0.0
    max_geometric_norm_error = 0.0
    total_windows = 0
    batches = 0
    with torch.inference_mode():
        for x, y in windows(validation, batch_size=32):
            captured.clear()
            logp = model.predict_log_probs(x)
            if len(captured) != 1:
                raise ValueError("Missing frozen expert capture")
            neural, count, weight = captured[0]
            if neural.shape != logp.shape:
                raise ValueError("Expert and scored distributions differ in shape")
            normal_error = max((neural.sum(-1) - 1).abs().max().item(),
                               (count.sum(-1) - 1).abs().max().item())
            max_expert_norm_error = max(max_expert_norm_error, normal_error)
            if normal_error > 3e-4:
                raise ValueError("Frozen expert distribution is not normalized")
            baseline = logp.exp()
            reconstructed = ((1 - weight).unsqueeze(-1) * neural
                             + weight.unsqueeze(-1) * count)
            error = (baseline - reconstructed).abs().max().item()
            max_reconstruction = max(max_reconstruction, error)
            if error > 3e-6:
                raise ValueError("Stage143 arithmetic mixture did not reconstruct")
            geometric = geometric_expert(neural, count, weight)
            geometric_error = (geometric.sum(-1) - 1).abs().max().item()
            max_geometric_norm_error = max(max_geometric_norm_error,
                                           geometric_error)
            if geometric_error > 2e-5:
                raise ValueError("Geometric product is not normalized")
            valid = y != -100
            target_ids = y.clamp_min(0).unsqueeze(-1)
            base_target = baseline.gather(-1, target_ids).squeeze(-1)[valid]
            geo_target = geometric.gather(-1, target_ids).squeeze(-1)[valid]
            current_windows = x.shape[0]
            if total_windows < EXPECTED_WINDOWS // 2 < total_windows + current_windows:
                raise ValueError("A batch crosses the fixed half-window boundary")
            half = 0 if total_windows < EXPECTED_WINDOWS // 2 else 1
            half_targets[half] += int(valid.sum())
            for beta in BETAS:
                target = (1 - beta) * base_target + beta * geo_target
                half_nll[half][beta] -= target.clamp_min(1e-30).log().double().sum().item()
            total_windows += current_windows
            batches += 1
            if batches % 8 == 0:
                print(f"validated {total_windows}/{EXPECTED_WINDOWS} windows", flush=True)
    if total_windows != EXPECTED_WINDOWS or sum(half_targets) != EXPECTED_TARGETS:
        raise ValueError("Incomplete validation coverage")
    scale = math.log(2) * byte_count
    baseline_bpb = sum(half_nll[half][0.0] for half in range(2)) / scale
    if abs(baseline_bpb - REFERENCE_BPB) > 2e-5:
        raise ValueError("Beta-zero score failed to reproduce Stage143")
    rows = []
    for beta in BETAS:
        parts = [half_nll[half][beta] for half in range(2)]
        nll = sum(parts)
        rows.append({
            "beta": beta,
            "bpb": nll / scale,
            "gain_bpb_vs_stage143": (sum(half_nll[half][0.0] for half in range(2)) - nll) / scale,
            "half_nll_nats": parts,
            "half_nll_improvement_nats": [half_nll[half][0.0] - parts[half]
                                           for half in range(2)],
        })
    best = min(rows[1:], key=lambda row: row["bpb"])
    advance = (best["gain_bpb_vs_stage143"] >= 0.015
               and all(value > 0 for value in best["half_nll_improvement_nats"]))
    result = {
        "protocol": PROTOCOL,
        "purpose": "validation_only_geometric_expert_fusion_screen",
        "split": "validation", "device": str(device), "precision": precision,
        "threads": args.threads, "checkpoint_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": implementation_sha,
        "source_sha256": sha(Path(__file__)), "graph_sha256": GRAPH_SHA256,
        "targets": sum(half_targets), "utf8_bytes": byte_count,
        "windows": total_windows, "batches": batches,
        "half_targets": half_targets, "reference_bpb": REFERENCE_BPB,
        "reconstructed_beta_zero_bpb": baseline_bpb,
        "maximum_arithmetic_reconstruction_error": max_reconstruction,
        "maximum_expert_normalization_error": max_expert_norm_error,
        "maximum_geometric_normalization_error": max_geometric_norm_error,
        "count_floor": 1e-12, "neural_floor": 1e-30,
        "grid": rows, "best_nonzero_beta": best["beta"],
        "advance_to_full_inference_qualification": advance,
        "no_test_scoring": True,
        "warning": "Validation-only screening; no trained or resource-qualified replacement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
