"""Fit a quadratic no-margin count gate on train text; validate once."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.build_kneser_ney import fit_kneser_ney
from scripts.fit_stage100_train_gate import (
    BASE_SHA, FEATURE_COLUMNS, collect_calibration, confidence_features5)
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump


INDICES = (0, 3, 5)  # neural maximum, highest backoff, highest row mass
RIDGE = (1e-4, 1e-3)
ANCHORS = (.05, .0625, .075)
SCALES = (.25, .5, .75, 1.0)
EXPECTED_FIXED_BPB = 1.401707662323582
MIN_WEIGHT = 1e-4


def basis(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 3:
        raise ValueError("Stage128 expects exactly three cheap features")
    a, b, c = x.unbind(-1)
    return torch.stack((a, b, c, a*a, b*b, c*c, a*b, a*c, b*c), -1)


def target_nll(neural: torch.Tensor, count: torch.Tensor,
               weight: torch.Tensor) -> torch.Tensor:
    return -torch.logaddexp(neural + torch.log1p(-weight),
                            count + weight.log())


def probabilities(x: torch.Tensor, coefficients: torch.Tensor,
                  anchor: float, slope: float) -> torch.Tensor:
    anchor_logit = math.log(anchor / (1 - anchor))
    raw = anchor_logit + slope * (x @ coefficients)
    return MIN_WEIGHT + (1 - 2 * MIN_WEIGHT) * torch.sigmoid(raw)


def fit(x: torch.Tensor, neural: torch.Tensor, count: torch.Tensor,
        penalty: float) -> torch.Tensor:
    coefficients = torch.zeros(x.shape[-1], dtype=torch.float64,
                               requires_grad=True)
    anchor_logit = math.log(.0625 / .9375)
    optimizer = torch.optim.LBFGS(
        [coefficients], lr=.5, max_iter=80, tolerance_grad=1e-9,
        tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        weight = MIN_WEIGHT + (1 - 2 * MIN_WEIGHT) * torch.sigmoid(
            anchor_logit + x @ coefficients)
        loss = target_nll(neural, count, weight).mean()
        loss = loss + penalty * coefficients.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return coefficients.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage128 evidence path")
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != BASE_SHA:
        raise ValueError("Unexpected frozen Stage92 or five-order count checkpoint")
    started = time.perf_counter()
    device, precision = setup("cpu", "fp32", 4)
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
    data = load_data()
    tokens = data["train"][0]
    cut90, cut95 = int(len(tokens) * .90), int(len(tokens) * .95)
    proxy, proxy_config, proxy_summary = fit_kneser_ney(
        tokens[:cut90].numpy(), max_order=5, min_count=2)
    proxy.eval()
    log_prior, prior_tokens = train_log_prior()
    fit_x, fit_neural, fit_count = collect_calibration(
        neural, proxy, tokens[cut90:cut95], log_prior)
    select_x, select_neural, select_count = collect_calibration(
        neural, proxy, tokens[cut95:], log_prior)

    fit_raw = fit_x[:, INDICES].double()
    mean, std = fit_raw.mean(0), fit_raw.std(0).clamp_min(1e-5)
    fit_normal = (fit_raw - mean) / std
    select_normal = (select_x[:, INDICES].double() - mean) / std
    fit_basis = basis(fit_normal)
    basis_mean = fit_basis.mean(0)
    basis_std = fit_basis.std(0).clamp_min(1e-5)
    fit_basis = (fit_basis - basis_mean) / basis_std
    select_basis = (basis(select_normal) - basis_mean) / basis_std
    candidates = []
    for penalty in RIDGE:
        coeff = fit(fit_basis, fit_neural.double(), fit_count.double(), penalty)
        for anchor in ANCHORS:
            for slope in SCALES:
                weight = probabilities(select_basis, coeff, anchor, slope)
                nll = float(target_nll(select_neural.double(),
                                       select_count.double(), weight).sum())
                candidates.append(dict(ridge=penalty, anchor=anchor,
                                       slope=slope, selection_nll=nll,
                                       selection_targets=len(select_neural),
                                       coefficients=coeff.tolist()))
    selected = min(candidates, key=lambda row: row["selection_nll"])
    chosen_coeff = torch.tensor(selected["coefficients"], dtype=torch.float64)
    validation, raw_bytes = data["validation"]
    fixed_nll = 0.0; gated_nll = 0.0; targets = 0
    with torch.inference_mode():
        for index, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            neural_logp = distilled_log_probs(neural, ids, log_prior)
            count_logp = counts.predict_log_probs(ids)
            a = neural_logp.gather(-1, target).squeeze(-1)[valid].double()
            b = count_logp.gather(-1, target).squeeze(-1)[valid].double()
            features = confidence_features5(
                neural_logp, count_logp, counts, ids)[valid.flatten()]
            raw = features[:, FEATURE_COLUMNS][:, INDICES].double()
            transformed = (basis((raw - mean) / std) - basis_mean) / basis_std
            gate = probabilities(transformed, chosen_coeff,
                                 selected["anchor"], selected["slope"])
            fixed_nll += float(target_nll(a, b, torch.full_like(a, .0625)).sum())
            gated_nll += float(target_nll(a, b, gate).sum())
            targets += int(valid.sum())
            if index % 10 == 0:
                print(json.dumps(dict(phase="validation", batch=index,
                                      targets=targets)), flush=True)
    fixed_bpb = fixed_nll / math.log(2) / raw_bytes
    gated_bpb = gated_nll / math.log(2) / raw_bytes
    if (targets != 376599 or raw_bytes != 1148007
            or abs(fixed_bpb - EXPECTED_FIXED_BPB) > 2e-5):
        raise ValueError("Static-mixture control or validation coverage mismatch")
    result = dict(
        protocol=PROTOCOL, purpose="stage128_train_fitted_quadratic_cheap_gate",
        split="validation", precision=precision,
        neural_sha256=NEURAL_SHA, counts_sha256=BASE_SHA,
        train_token_count=len(tokens), count_fit_end=cut90, gate_fit_end=cut95,
        gate_fit_targets=len(fit_neural), gate_selection_targets=len(select_neural),
        proxy_count_config=proxy_config, proxy_count_summary=proxy_summary,
        raw_feature_indices=list(INDICES), raw_feature_mean=mean.tolist(),
        raw_feature_std=std.tolist(), basis_mean=basis_mean.tolist(),
        basis_std=basis_std.tolist(), train_selection_candidates=candidates,
        train_selected_recipe=selected,
        fixed_validation_bpb=fixed_bpb, gated_validation_bpb=gated_bpb,
        improvement_over_fixed_bpb=fixed_bpb-gated_bpb,
        improvement_over_stage117_best_bpb=1.400478201-gated_bpb,
        below_1_4_diagnostic=gated_bpb < 1.4,
        targets=targets, utf8_bytes=raw_bytes,
        train_unigram_tokens=prior_tokens,
        seconds=time.perf_counter() - started,
        gate_coefficients_fit_train_only=True,
        scalar_selection_train_only=True,
        validation_scored_once=True, exported_checkpoint=False,
        no_test_scoring=True, source_sha256=sha(Path(__file__)))
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"train_selection_candidates": [],
                               "proxy_count_summary": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
