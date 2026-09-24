"""Refit a cheaper gate on held-out training text, then screen fixed scalars."""
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

import analyze_stage60_deployable_gate as stage60
from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.build_kneser_ney import fit_kneser_ney
from scripts.fit_stage100_train_gate import (
    BASE_SHA, FEATURE_COLUMNS, FEATURE_NAMES, collect_calibration,
    confidence_features5)
from scripts.screen_stage110_low_cost_gate import STAGE100_SHA
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump


INDICES_FULL = (0, 1, 3, 5)
INDICES_CHEAP = (0, 3, 5)
ANCHORS = (.05, .0625, .075, .09, .11)
SCALES = (.2, .3, .4, .5, .6, .7)
EXPECTED_STAGE114_NOMARGIN = 1.4005819056954032


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--stage114", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage117 output path")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != BASE_SHA
            or sha(args.train_gate) != STAGE100_SHA):
        raise ValueError("Unexpected frozen expert or Stage100 gate")
    train_gate = json.loads(args.train_gate.read_text(encoding="utf-8-sig"))
    stage114 = json.loads(args.stage114.read_text(encoding="utf-8-sig"))
    stage114_control = next(
        row for row in stage114["candidates"]
        if row["variant"] == "no_margin" and row["slope_scale"] == .4)
    if (train_gate.get("no_validation_or_test_fitting") is not True
            or train_gate.get("feature_names") != list(FEATURE_NAMES)
            or stage114.get("protocol") != PROTOCOL
            or stage114.get("neural_sha256") != NEURAL_SHA
            or stage114.get("counts_sha256") != BASE_SHA
            or abs(stage114_control["bpb"] - EXPECTED_STAGE114_NOMARGIN) > 1e-10):
        raise ValueError("Unexpected train gate or Stage114 reference")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    neural.eval(); counts.eval()
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
    all_features = torch.cat((fit_features, select_features))
    all_neural = torch.cat((fit_neural, select_neural))
    all_count = torch.cat((fit_count, select_count))
    fit_mask = torch.arange(len(all_features)) < len(fit_features)
    select_mask = ~fit_mask
    cheap_names = tuple(FEATURE_NAMES[i] for i in INDICES_CHEAP)
    fitted = stage60.fit_gate(
        all_features[:, INDICES_CHEAP], cheap_names, all_neural, all_count,
        fit_mask, select_mask)
    fitted_mean = fit_features[:, INDICES_CHEAP].double().mean(0)
    fitted_std = fit_features[:, INDICES_CHEAP].double().std(0).clamp_min(1e-5)
    old_mean = torch.tensor(train_gate["feature_mean"], dtype=torch.float64)
    old_std = torch.tensor(train_gate["feature_std"], dtype=torch.float64)
    old_coeff = torch.tensor(
        [train_gate["coefficients"][name] for name in FEATURE_NAMES],
        dtype=torch.float64)
    recipes = {
        "old_full": (INDICES_FULL, old_mean[list(INDICES_FULL)],
                     old_std[list(INDICES_FULL)], old_coeff[list(INDICES_FULL)]),
        "old_no_margin": (INDICES_CHEAP, old_mean[list(INDICES_CHEAP)],
                          old_std[list(INDICES_CHEAP)], old_coeff[list(INDICES_CHEAP)]),
        "refit_no_margin": (INDICES_CHEAP, fitted_mean, fitted_std,
                            torch.tensor([fitted["coefficients"][name]
                                          for name in cheap_names], dtype=torch.float64)),
    }
    validation, raw_bytes = data["validation"]
    anchor_logits = torch.tensor(
        [math.log(a / (1 - a)) for a in ANCHORS], dtype=torch.float64)
    slopes = torch.tensor(SCALES, dtype=torch.float64)
    totals = torch.zeros((len(recipes), len(ANCHORS), len(SCALES)),
                         dtype=torch.float64)
    weight_sums = torch.zeros_like(totals)
    targets = 0
    with torch.inference_mode():
        for batch, (ids, labels) in enumerate(windows(validation, 32)):
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            neural_logp = distilled_log_probs(neural, ids, log_prior)
            count_logp = counts.predict_log_probs(ids)
            neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid].double()
            count_target = count_logp.gather(-1, target).squeeze(-1)[valid].double()
            feature = confidence_features5(
                neural_logp, count_logp, counts, ids)[valid.flatten(), :]
            feature = feature[:, FEATURE_COLUMNS].double()
            for index, (columns, mean, std, coeff) in enumerate(recipes.values()):
                slope = ((feature[:, columns] - mean) / std * coeff).sum(-1)
                logits = (anchor_logits[None, :, None]
                          + slope[:, None, None] * slopes[None, None, :])
                weight = torch.sigmoid(logits)
                weight = stage60.MIN_WEIGHT + (1 - 2 * stage60.MIN_WEIGHT) * weight
                totals[index] += stage60.mixture_nll(
                    neural_target[:, None, None],
                    count_target[:, None, None], weight).sum(0)
                weight_sums[index] += weight.sum(0)
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(phase="validation", batch=batch,
                                      targets=targets)), flush=True)
    rows = [dict(recipe=name, active_features=[FEATURE_NAMES[k] for k in recipe[0]],
                 anchor_weight=anchor, slope_scale=scale,
                 mean_count_weight=float(weight_sums[i, j, k] / targets),
                 nll_nats=float(totals[i, j, k]),
                 bpb=float(totals[i, j, k] / math.log(2) / raw_bytes))
            for i, (name, recipe) in enumerate(recipes.items())
            for j, anchor in enumerate(ANCHORS)
            for k, scale in enumerate(SCALES)]
    control = next(row for row in rows if row["recipe"] == "old_no_margin"
                   and row["anchor_weight"] == .0625 and row["slope_scale"] == .4)
    if (targets != 376599 or raw_bytes != 1148007
            or abs(control["bpb"] - EXPECTED_STAGE114_NOMARGIN) > 2e-6):
        raise ValueError("Stage114 no-margin control or coverage mismatch")
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="train_only_no_margin_refit_and_bounded_anchor_screen",
        neural_sha256=NEURAL_SHA, counts_sha256=BASE_SHA,
        old_gate_sha256=STAGE100_SHA, stage114_sha256=sha(args.stage114),
        train_token_count=len(train_tokens), count_fit_end=cut90,
        gate_fit_end=cut95, gate_fit_targets=len(fit_neural),
        gate_selection_targets=len(select_neural),
        proxy_count_config=proxy_config, proxy_count_summary=proxy_summary,
        no_margin_train_fit=dict(mean=fitted_mean.tolist(), std=fitted_std.tolist(),
                                 **fitted),
        anchors=ANCHORS, scales=SCALES, candidates=rows,
        reference=control, best=min(rows, key=lambda row: row["bpb"]),
        best_no_margin=min((row for row in rows if "no_margin" in row["recipe"]),
                           key=lambda row: row["bpb"]),
        sub_1_4_candidates=[row for row in rows if row["bpb"] < 1.4],
        targets=targets, utf8_bytes=raw_bytes,
        train_unigram_tokens=train_unigram_tokens,
        seconds=time.perf_counter() - started,
        train_only_coefficients=True, validation_selected_scalars_only=True,
        no_test_scoring=True, exported_checkpoint=False,
        source_sha256=sha(Path(__file__)))
    atomic_json_dump(result, args.output)
    print(json.dumps(result | {"candidates": [], "sub_1_4_candidates": [],
                                "proxy_count_summary": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
