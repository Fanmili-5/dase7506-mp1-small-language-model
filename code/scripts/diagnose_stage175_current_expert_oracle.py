"""Hindsight-only target oracle for the unchanged Stage143 neural/count pair."""
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
REFERENCE_BPB = 1.399686162042141
EXPECTED_TARGETS = 376_599
EXPECTED_BYTES = 1_148_007


def recover_target_probabilities(
        neural_weighted: torch.Tensor,
        mixture: torch.Tensor,
        count_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo a known input-only convex mixture at the observed target only.

    This diagnostic reads validation labels. It must never be an inference
    method or a checkpoint parameter fit.
    """
    if (neural_weighted.shape != mixture.shape
            or mixture.shape != count_weight.shape
            or not torch.isfinite(neural_weighted).all()
            or not torch.isfinite(mixture).all()
            or not torch.isfinite(count_weight).all()
            or not ((count_weight > 0) & (count_weight < 1)).all()):
        raise ValueError("Invalid paired target mixture records")
    neural = neural_weighted.double() / (1 - count_weight.double())
    count = (mixture.double() - neural_weighted.double()) / count_weight.double()
    if (not torch.isfinite(neural).all() or not torch.isfinite(count).all()
            or (neural <= 0).any() or (neural > 1.0001).any()
            or (count < -1e-6).any() or (count > 1.0001).any()):
        raise ValueError(
            "Expert target probabilities are invalid: "
            f"neural=[{neural.min().item():.8g},{neural.max().item():.8g}], "
            f"count=[{count.min().item():.8g},{count.max().item():.8g}], "
            f"weight=[{count_weight.min().item():.8g},{count_weight.max().item():.8g}]")
    return neural, count.clamp_min(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("Use a new output file and a positive thread count")
    if sha(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Stage143 checkpoint changed")
    device, precision = setup("cpu", "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("implementation") != "student_stage143_openvino_singlepass"):
        raise ValueError("Unexpected checkpoint protocol or implementation")
    model, implementation_sha = make_model(checkpoint["implementation"],
                                            checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    validation, byte_count = load_data()["validation"]
    if byte_count != EXPECTED_BYTES:
        raise ValueError("Validation raw-byte coverage changed")

    original_add = model.ngram.add_collected
    batch_targets: torch.Tensor | None = None
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_add(result, weight, terms, suffix):
        if batch_targets is None:
            raise ValueError("Missing diagnostic target batch")
        target_ids = batch_targets.clamp_min(0).unsqueeze(-1)
        before = result.gather(-1, target_ids).squeeze(-1).clone()
        returned = original_add(result, weight, terms, suffix)
        after = returned.gather(-1, target_ids).squeeze(-1).clone()
        captured.append((before, after, weight.clone()))
        return returned

    model.ngram.add_collected = capture_add
    total_reference_nll = 0.0
    total_reconstruction_nll = 0.0
    total_oracle_nll = 0.0
    oracle_count_better = 0
    targets = 0
    batches = 0
    max_reconstruction_difference = 0.0
    with torch.inference_mode():
        for x, y in windows(validation, batch_size=32):
            batch_targets = y
            captured.clear()
            logp = model.predict_log_probs(x)
            if len(captured) != 1:
                raise ValueError("Expected exactly one sparse addition per batch")
            before, after, weight = captured[0]
            valid = y != -100
            observed_logp = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid]
            neural, count = recover_target_probabilities(before[valid], after[valid], weight[valid])
            reconstructed = (1 - weight[valid].double()) * neural + weight[valid].double() * count
            maximum = torch.maximum(neural, count)
            if (reconstructed <= 0).any() or (maximum <= 0).any():
                raise ValueError("Non-positive target probability")
            difference = (reconstructed.float() - observed_logp.exp()).abs().max().item()
            max_reconstruction_difference = max(max_reconstruction_difference, difference)
            if difference > 2e-6:
                raise ValueError("Stage143 mixture cannot be reconstructed")
            total_reference_nll -= observed_logp.double().sum().item()
            total_reconstruction_nll -= reconstructed.log().sum().item()
            total_oracle_nll -= maximum.log().sum().item()
            oracle_count_better += int((count > neural).sum())
            targets += int(valid.sum())
            batches += 1
    if targets != EXPECTED_TARGETS:
        raise ValueError("Incomplete validation target coverage")
    scale = math.log(2) * byte_count
    reference_bpb = total_reference_nll / scale
    reconstruction_bpb = total_reconstruction_nll / scale
    oracle_bpb = total_oracle_nll / scale
    if (abs(reference_bpb - REFERENCE_BPB) > 2e-5
            or abs(reconstruction_bpb - reference_bpb) > 2e-5
            or oracle_bpb > reconstruction_bpb + 1e-8):
        raise ValueError("Frozen score identity or oracle ordering failed")
    result = {
        "protocol": PROTOCOL,
        "purpose": "hindsight_current_expert_oracle_not_deployable",
        "split": "validation",
        "device": str(device),
        "precision": precision,
        "threads": args.threads,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": implementation_sha,
        "source_sha256": sha(Path(__file__)),
        "graph_sha256": sha(ROOT / "inference_assets/stage143-stage92-features.onnx"),
        "targets": targets,
        "utf8_bytes": byte_count,
        "batches": batches,
        "reference_stage143_bpb": reference_bpb,
        "reconstructed_stage143_bpb": reconstruction_bpb,
        "oracle_bpb": oracle_bpb,
        "oracle_gain_bpb": reference_bpb - oracle_bpb,
        "oracle_count_better_targets": oracle_count_better,
        "maximum_target_probability_reconstruction_difference": max_reconstruction_difference,
        "oracle_below_1_35": oracle_bpb < 1.35,
        "no_test_scoring": True,
        "warning": "Uses the true validation target to choose an expert; this is impossible at inference and is not a score or a valid gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
