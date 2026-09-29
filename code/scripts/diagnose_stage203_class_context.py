"""Fixed train-derived class-context correction on complete validation only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from common import PROTOCOL, make_model, setup, sha, windows
from scripts.train_stage193_fresh_mixture_pilot import load_train_validation

CHECKPOINT_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
GRAPH = ROOT / "inference_assets/stage143-stage92-features.onnx"
GRAPH_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
PLAN = ROOT / "docs/STAGE203_CLASS_CONTEXT_EXPERT_PLAN_20260928.md"
CLASSES = 64
SEED = 203017
LLOYD_STEPS = 20
BIGRAM_PRIOR = 32.0
TRIGRAM_PRIOR = 64.0
GAMMA = 0.25
REFERENCE_BPB = 1.399686162042141
TARGETS = 376599
BYTES = 1148007
HALF_WINDOWS = 736
REQUIRED_GAIN = 0.03
ASSET_MARGIN = 11298452


def train_clusters(embeddings: np.ndarray, frequency: np.ndarray) -> tuple[np.ndarray, dict]:
    """Deterministic cosine Lloyd clustering of frozen train-derived token rows."""
    if embeddings.shape != (2048, 288) or frequency.shape != (2048,):
        raise ValueError("Unexpected tied output matrix or train frequency shape")
    rows = embeddings.astype(np.float64, copy=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Invalid output embedding norm")
    rows /= norms
    rng = np.random.default_rng(SEED)
    probabilities = (frequency.astype(np.float64) + 1)
    probabilities /= probabilities.sum()
    initial = rng.choice(len(rows), size=CLASSES, replace=False, p=probabilities)
    centroids = rows[initial].copy()
    for _ in range(LLOYD_STEPS):
        labels = np.argmax(rows @ centroids.T, axis=1)
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, rows)
        counts = np.bincount(labels, minlength=CLASSES)
        nonempty = counts > 0
        new_norms = np.linalg.norm(sums[nonempty], axis=1, keepdims=True)
        centroids[nonempty] = sums[nonempty] / new_norms
    labels = np.argmax(rows @ centroids.T, axis=1).astype(np.int64)
    sizes = np.bincount(labels, minlength=CLASSES)
    if np.any(sizes == 0):
        raise ValueError("Empty final token class")
    return labels, {
        "initial_token_ids": initial.tolist(),
        "min_class_tokens": int(sizes.min()),
        "max_class_tokens": int(sizes.max()),
        "class_token_sizes": sizes.tolist(),
    }


def class_transition_tables(train_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalized hierarchical unigram/bigram/trigram from train classes."""
    if train_classes.ndim != 1 or len(train_classes) < 3:
        raise ValueError("Need at least three training tokens")
    if np.any((train_classes < 0) | (train_classes >= CLASSES)):
        raise ValueError("Invalid class ID")
    uni = np.bincount(train_classes, minlength=CLASSES).astype(np.float64)
    bi = np.zeros((CLASSES, CLASSES), dtype=np.float64)
    tri = np.zeros((CLASSES, CLASSES, CLASSES), dtype=np.float64)
    np.add.at(bi, (train_classes[:-1], train_classes[1:]), 1)
    np.add.at(tri, (train_classes[:-2], train_classes[1:-1],
                    train_classes[2:]), 1)
    p_uni = (uni + 1) / (uni.sum() + CLASSES)
    p_bi = (bi + BIGRAM_PRIOR * p_uni[None, :]) / (
        bi.sum(axis=-1, keepdims=True) + BIGRAM_PRIOR)
    p_tri = (tri + TRIGRAM_PRIOR * p_bi[None, :, :]) / (
        tri.sum(axis=-1, keepdims=True) + TRIGRAM_PRIOR)
    if (np.max(np.abs(p_uni.sum() - 1)) > 1e-12
            or np.max(np.abs(p_bi.sum(axis=-1) - 1)) > 1e-12
            or np.max(np.abs(p_tri.sum(axis=-1) - 1)) > 1e-12
            or np.any(p_tri <= 0)):
        raise ValueError("Class transition rows did not normalize")
    return p_bi, p_tri, {
        "train_unigram_observations": int(uni.sum()),
        "train_bigram_observations": int(bi.sum()),
        "train_trigram_observations": int(tri.sum()),
        "observed_trigram_cells": int(np.count_nonzero(tri)),
        "train_class_frequency_min": int(uni.min()),
        "train_class_frequency_max": int(uni.max()),
    }


def next_class_probabilities(ids: torch.Tensor, labels: torch.Tensor,
                             bigram: torch.Tensor, trigram: torch.Tensor) -> torch.Tensor:
    """Use only observed current/prior input tokens within each row."""
    classes = labels[ids]
    next_probs = torch.empty((*ids.shape, CLASSES), dtype=torch.float64)
    next_probs[:, 0] = bigram[classes[:, 0]]
    if ids.shape[1] > 1:
        next_probs[:, 1:] = trigram[classes[:, :-1], classes[:, 1:]]
    return next_probs


def corrected_log_probs(logp: torch.Tensor, labels: torch.Tensor,
                        next_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Full normalized geometric correction; never reads next-token labels."""
    if logp.ndim != 3 or logp.shape[-1] != 2048:
        raise ValueError("Expected [batch,time,2048] base log probabilities")
    if next_probs.shape != (*logp.shape[:2], CLASSES):
        raise ValueError("Invalid class probability shape")
    base = logp.double().exp()
    expanded = labels.view(1, 1, -1).expand_as(base)
    mass = torch.zeros((*logp.shape[:2], CLASSES), dtype=torch.float64)
    mass.scatter_add_(-1, expanded, base)
    if not torch.isfinite(mass).all() or torch.any(mass <= 0):
        raise ValueError("Invalid base class masses")
    ratio = (next_probs / mass).pow(GAMMA)
    normalizer = (mass * ratio).sum(dim=-1, keepdim=True)
    if not torch.isfinite(normalizer).all() or torch.any(normalizer <= 0):
        raise ValueError("Invalid geometric normalizer")
    corrected = logp.double() + ratio.log().gather(
        -1, expanded) - normalizer.log()
    return corrected, mass


def synthetic_causality_check(model: torch.nn.Module, labels: torch.Tensor,
                              bigram: torch.Tensor, trigram: torch.Tensor) -> dict:
    """A future-input change cannot affect earlier or other-row predictions."""
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, 2048, (2, 256), generator=generator)
    altered = ids.clone()
    altered[0, 128] = (altered[0, 128] + 1) % 2048

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        logp = model.predict_log_probs(inputs).float()
        q = next_class_probabilities(inputs, labels, bigram, trigram)
        corrected, _ = corrected_log_probs(logp, labels, q)
        return corrected

    original = predict(ids)
    changed = predict(altered)
    prefix_error = float((original[0, :128] - changed[0, :128]).abs().max())
    other_row_error = float((original[1] - changed[1]).abs().max())
    normalization_error = max(float(original.logsumexp(-1).abs().max()),
                              float(changed.logsumexp(-1).abs().max()))
    if (prefix_error > 2e-5 or other_row_error > 2e-5
            or normalization_error > 2e-5):
        raise ValueError("Synthetic causal/normalization check failed")
    return {"prefix_max_logp_error": prefix_error,
            "other_row_max_logp_error": other_row_error,
            "normalization_max_error": normalization_error}


def score_candidate(checkpoint: Path) -> dict:
    if sha(checkpoint) != CHECKPOINT_SHA or sha(GRAPH) != GRAPH_SHA:
        raise ValueError("Frozen Stage143 checkpoint or graph changed")
    if not PLAN.is_file():
        raise ValueError("Missing pre-outcome Stage203 plan")
    data = load_train_validation()
    train_tokens = data["train"][0].numpy()
    validation_tokens, byte_count = data["validation"]
    if len(validation_tokens) - 1 != TARGETS or byte_count != BYTES:
        raise ValueError("Unexpected complete-validation coverage")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_stage143_openvino_singlepass"):
        raise ValueError("Unexpected frozen checkpoint contract")
    output_rows = payload["model"]["neural.head.weight"].numpy()
    frequency = np.bincount(train_tokens, minlength=2048)
    labels_np, clustering = train_clusters(output_rows, frequency)
    train_classes = labels_np[train_tokens]
    p_bi, p_tri, counts = class_transition_tables(train_classes)
    labels = torch.from_numpy(labels_np)
    bigram = torch.from_numpy(p_bi)
    trigram = torch.from_numpy(p_tri)
    device, precision = setup("cpu", "fp32", 4)
    model, implementation_sha = make_model(payload["implementation"],
                                            payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        synthetic_checks = synthetic_causality_check(model, labels, bigram, trigram)
    baseline_nll = 0.0
    candidate_nll = 0.0
    gain_by_half = [0.0, 0.0]
    total_targets = 0
    window_number = 0
    max_normalization_error = 0.0
    with torch.inference_mode():
        for ids, targets in windows(validation_tokens, batch_size=32):
            logp = model.predict_log_probs(ids).float()
            if (logp.shape != (*ids.shape, 2048)
                    or not torch.isfinite(logp).all()
                    or float(logp.logsumexp(-1).abs().max()) > 1e-3):
                raise ValueError("Frozen base output invalid")
            next_probs = next_class_probabilities(ids, labels, bigram, trigram)
            corrected, _ = corrected_log_probs(logp, labels, next_probs)
            if window_number == 0:
                max_normalization_error = float(corrected.logsumexp(-1).abs().max())
                if max_normalization_error > 2e-5:
                    raise ValueError("Class correction is not normalized")
            mask = targets != -100
            safe_targets = targets.clamp_min(0).unsqueeze(-1)
            base_loss = -logp.gather(-1, safe_targets).squeeze(-1).double()
            candidate_loss = -corrected.gather(-1, safe_targets).squeeze(-1)
            base_loss = base_loss.masked_fill(~mask, 0)
            candidate_loss = candidate_loss.masked_fill(~mask, 0)
            base_rows = base_loss.sum(-1).tolist()
            candidate_rows = candidate_loss.sum(-1).tolist()
            for base_row, candidate_row in zip(base_rows, candidate_rows):
                baseline_nll += base_row
                candidate_nll += candidate_row
                half = int(window_number >= HALF_WINDOWS)
                gain_by_half[half] += base_row - candidate_row
                window_number += 1
            total_targets += int(mask.sum())
    if total_targets != TARGETS or window_number != 1472:
        raise ValueError("Incomplete independent-window validation")
    baseline_bpb = baseline_nll / math.log(2) / BYTES
    candidate_bpb = candidate_nll / math.log(2) / BYTES
    if abs(baseline_bpb - REFERENCE_BPB) > 2e-5:
        raise ValueError("Frozen Stage143 score did not reproduce")
    gain = baseline_bpb - candidate_bpb
    extra_bytes = 4 * CLASSES**3 + 4 * CLASSES**2 + 2 * 2048
    return {
        "status": "diagnostic_only_not_deployable",
        "protocol": PROTOCOL, "split": "validation", "device": "cpu",
        "precision": precision, "threads": 4,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "graph_sha256": GRAPH_SHA,
        "implementation_sha256": implementation_sha,
        "source_sha256": sha(Path(__file__)), "plan_sha256": sha(PLAN),
        "train_sha256": sha(ROOT / "data/wikitext_train.txt"),
        "validation_sha256": sha(ROOT / "data/wikitext_validation.txt"),
        "tokenizer_sha256": sha(ROOT / "data/tokenizer.json"),
        "train_tokens": len(train_tokens), "validation_targets": total_targets,
        "validation_utf8_bytes": BYTES, "independent_windows": window_number,
        "classes": CLASSES, "seed": SEED, "lloyd_steps": LLOYD_STEPS,
        "bigram_prior": BIGRAM_PRIOR, "trigram_prior": TRIGRAM_PRIOR,
        "gamma": GAMMA, "clustering": clustering, "class_counts": counts,
        "synthetic_checks": synthetic_checks,
        "baseline_bpb": baseline_bpb, "candidate_bpb": candidate_bpb,
        "gain_bpb": gain, "gain_nats_by_half": gain_by_half,
        "required_gain_bpb": REQUIRED_GAIN,
        "projected_extra_fp32_asset_bytes": extra_bytes,
        "asset_margin_bytes": ASSET_MARGIN,
        "asset_screen_passed": extra_bytes <= ASSET_MARGIN,
        "max_first_batch_normalization_error": max_normalization_error,
        "elapsed_seconds": time.perf_counter() - start,
        "advance_to_cpu_implementation": bool(gain >= REQUIRED_GAIN
                                              and all(value > 0 for value in gain_by_half)
                                              and extra_bytes <= ASSET_MARGIN),
        "test_text_opened_by_this_script": False,
        "test_scored_by_this_script": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new diagnostic output")
    result = score_candidate(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
