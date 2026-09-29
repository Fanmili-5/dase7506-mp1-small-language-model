"""Fit a causal neural/MKN gate on training text and score validation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from torch.nn import functional as F

import analyze_stage50_confidence_gate as stage50
import analyze_stage60_deployable_gate as stage60
from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import (
    NEURAL_SHA, distilled_log_probs)
from scripts.build_kneser_ney import fit_kneser_ney
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump


BASE_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
EXTENDED_SHA = "b901f11766c005733f769fa5582e82f586db31a41edaefe93122aae1ed471953"
FEATURE_COLUMNS = stage60.FEATURE_SETS["neural_top2_sparse_prefix"]
FEATURE_NAMES = tuple(stage50.FEATURE_NAMES[index] for index in FEATURE_COLUMNS)
REFERENCE = {"order5": 1.401707662323582, "order6": 1.4012884354781747}


def confidence_features5(neural_logp, count_logp, count_model, ids):
    """Keep the Stage89 feature layout; map order-6 prefixes into order-5 bin."""
    neural_top = neural_logp.topk(2, dim=-1)
    count_top = count_logp.topk(2, dim=-1)
    neural_prob, count_prob = neural_logp.exp(), count_logp.exp()
    entropy_scale = math.log(neural_logp.shape[-1])
    neural_entropy = -(neural_prob * neural_logp).sum(-1) / entropy_scale
    count_entropy = -(count_prob * count_logp).sum(-1) / entropy_scale
    neural_at_count = neural_logp.gather(-1, count_top.indices[..., :1]).squeeze(-1)
    count_at_neural = count_logp.gather(-1, neural_top.indices[..., :1]).squeeze(-1)
    order, backoff, edges, max_mass, total_mass = stage50.count_context_features(
        count_model, ids)
    position = torch.arange(ids.shape[1]).expand_as(ids).float() / max(1, ids.shape[1] - 1)
    one_hot_order = F.one_hot(order.clamp_max(5) - 1, num_classes=5).float()
    flat = lambda value: value.reshape(-1).float()
    columns = (
        flat(neural_top.values[..., 0]), flat(count_top.values[..., 0]),
        flat(neural_top.values[..., 0] - neural_top.values[..., 1]),
        flat(count_top.values[..., 0] - count_top.values[..., 1]),
        flat(neural_entropy), flat(count_entropy),
        flat(neural_top.indices[..., 0] == count_top.indices[..., 0]),
        flat(neural_at_count), flat(count_at_neural), flat(position),
        backoff, torch.log1p(edges), max_mass, total_mass,
    )
    return torch.cat((torch.stack(columns, dim=1), one_hot_order), dim=1)


def collect_calibration(neural, counts, tokens, log_prior):
    feature_batches, neural_batches, count_batches = [], [], []
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(tokens, 32)):
            valid = labels != -100
            neural_logp = distilled_log_probs(neural, ids, log_prior)
            count_logp = counts.predict_log_probs(ids)
            features = confidence_features5(neural_logp, count_logp, counts, ids)
            targets = labels.clamp_min(0).unsqueeze(-1)
            feature_batches.append(features[valid.flatten(), :][:, FEATURE_COLUMNS])
            neural_batches.append(neural_logp.gather(-1, targets).squeeze(-1)[valid])
            count_batches.append(count_logp.gather(-1, targets).squeeze(-1)[valid])
            if batch % 10 == 0:
                print(json.dumps(dict(phase="train_calibration", batch=batch)), flush=True)
    return (torch.cat(feature_batches), torch.cat(neural_batches),
            torch.cat(count_batches))


def gate_weight(features, mean, std, coefficients, bias):
    standardized = (features.double() - mean) / std
    weight = torch.sigmoid(standardized @ coefficients + bias)
    return stage60.MIN_WEIGHT + (1 - 2 * stage60.MIN_WEIGHT) * weight


def score_validation(neural, counts, tokens, raw_bytes, log_prior,
                     mean, std, coefficients, bias, label):
    fixed_nll = 0.; gated_nll = 0.; targets = 0; weights = []
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(tokens, 32)):
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            neural_logp = distilled_log_probs(neural, ids, log_prior)
            count_logp = counts.predict_log_probs(ids)
            features = confidence_features5(neural_logp, count_logp, counts, ids)
            features = features[valid.flatten(), :][:, FEATURE_COLUMNS]
            neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid].double()
            count_target = count_logp.gather(-1, target).squeeze(-1)[valid].double()
            weight = gate_weight(features, mean, std, coefficients, bias)
            fixed = torch.full_like(weight, .0625)
            fixed_nll += float(stage60.mixture_nll(
                neural_target, count_target, fixed).sum())
            gated_nll += float(stage60.mixture_nll(
                neural_target, count_target, weight).sum())
            weights.append(weight.float())
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(phase=label, batch=batch, targets=targets)),
                      flush=True)
    all_weights = torch.cat(weights)
    result = dict(expert=label, fixed_weight_bpb=fixed_nll / math.log(2) / raw_bytes,
                  train_gate_bpb=gated_nll / math.log(2) / raw_bytes,
                  gate_gain_bpb=(fixed_nll - gated_nll) / math.log(2) / raw_bytes,
                  mean_weight=float(all_weights.mean()),
                  weight_p10=float(all_weights.quantile(.1)),
                  weight_p50=float(all_weights.quantile(.5)),
                  weight_p90=float(all_weights.quantile(.9)),
                  targets=targets, utf8_bytes=raw_bytes)
    if (targets != 376599 or raw_bytes != 1148007
            or abs(result["fixed_weight_bpb"] - REFERENCE[label]) > 2e-5):
        raise ValueError("Fixed mixture or validation coverage mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if (sha(args.neural) != NEURAL_SHA or sha(args.base) != BASE_SHA
            or sha(args.extended) != EXTENDED_SHA):
        raise ValueError("Unexpected pinned checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    base_payload = torch.load(args.base, map_location="cpu", weights_only=True)
    extended_payload = torch.load(args.extended, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or base_payload.get("implementation") != "student_ngram"
            or extended_payload.get("implementation") != "student_ngram"
            or extended_payload["config"].get("max_order") != 6):
        raise ValueError("Unexpected expert payload")
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True); neural.eval()
    full_counts = {}
    for label, payload in (("order5", base_payload), ("order6", extended_payload)):
        model, _ = make_model("student_ngram", payload["config"], device)
        model.load_state_dict(payload["model"], strict=True); model.eval()
        full_counts[label] = model
    data = load_data()
    train_tokens = data["train"][0]
    cut90, cut95 = int(len(train_tokens) * .90), int(len(train_tokens) * .95)
    started = time.perf_counter()
    proxy, proxy_config, proxy_summary = fit_kneser_ney(
        train_tokens[:cut90].numpy(), max_order=5, min_count=2)
    proxy.eval()
    log_prior, train_unigram_tokens = train_log_prior()
    fit_features, fit_neural, fit_count = collect_calibration(
        neural, proxy, train_tokens[cut90:cut95], log_prior)
    select_features, select_neural, select_count = collect_calibration(
        neural, proxy, train_tokens[cut95:], log_prior)
    features = torch.cat((fit_features, select_features))
    neural_target = torch.cat((fit_neural, select_neural))
    count_target = torch.cat((fit_count, select_count))
    fit_mask = torch.arange(len(features)) < len(fit_features)
    select_mask = ~fit_mask
    selected = stage60.fit_gate(features, FEATURE_NAMES, neural_target,
                                count_target, fit_mask, select_mask)
    mean = fit_features.double().mean(0)
    std = fit_features.double().std(0).clamp_min(1e-5)
    coefficients = torch.tensor([selected["coefficients"][name]
                                 for name in FEATURE_NAMES], dtype=torch.float64)
    bias = selected["bias"]
    selected_fixed = stage60.mixture_nll(
        select_neural.double(), select_count.double(),
        torch.full_like(select_neural.double(), .0625)).sum()
    train_selection = dict(gated_mean_nll=selected["nll_nats"] / selected["targets"],
                           fixed_mean_nll=float(selected_fixed) / len(select_neural),
                           targets=int(selected["targets"]))
    validation, raw_bytes = data["validation"]
    results = {label: score_validation(neural, counts, validation, raw_bytes,
                                       log_prior, mean, std, coefficients, bias,
                                       label)
               for label, counts in full_counts.items()}
    result = dict(
        protocol=PROTOCOL, purpose="train_only_gate_fit_validation_screen",
        neural_sha256=NEURAL_SHA, base_counts_sha256=BASE_SHA,
        extended_counts_sha256=EXTENDED_SHA,
        train_token_count=len(train_tokens), count_fit_end=cut90,
        gate_fit_end=cut95, gate_fit_targets=len(fit_neural),
        gate_selection_targets=len(select_neural),
        proxy_count_config=proxy_config, proxy_count_summary=proxy_summary,
        feature_names=FEATURE_NAMES, feature_mean=mean.tolist(),
        feature_std=std.tolist(), coefficients=selected["coefficients"],
        bias=bias, train_selection=train_selection,
        validation_results=results, train_unigram_tokens=train_unigram_tokens,
        seconds=time.perf_counter() - started,
        neural_saw_gate_training_segments=True,
        proxy_count_excluded_gate_segments=True,
        no_validation_or_test_fitting=True, no_test_scoring=True,
        exported_checkpoint=False, source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"proxy_count_summary": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
