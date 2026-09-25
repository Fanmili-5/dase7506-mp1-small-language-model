"""Out-of-half, validation-fitted Stage143 gate diagnostic; never deploy."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from diagnose_stage175_current_expert_oracle import (
    CHECKPOINT_SHA256, EXPECTED_BYTES, EXPECTED_TARGETS, REFERENCE_BPB,
    recover_target_probabilities,
)

SEED = 176017
EPOCHS = 20
BATCH_SIZE = 8192
HIDDEN_WIDTH = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
GATE_FLOOR = 1e-4


class ResidualGate(nn.Module):
    """Zero-start correction to the existing causal Stage143 gate."""

    def __init__(self, feature_count: int):
        super().__init__()
        self.first = nn.Linear(feature_count, HIDDEN_WIDTH)
        self.last = nn.Linear(HIDDEN_WIDTH, 1)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, standardized: torch.Tensor,
                original_weight: torch.Tensor) -> torch.Tensor:
        clipped = ((original_weight - GATE_FLOOR) / (1 - 2 * GATE_FLOOR))
        clipped = clipped.clamp(1e-6, 1 - 1e-6)
        base = torch.logit(clipped)
        delta = 5 * torch.tanh(self.last(torch.tanh(self.first(standardized))).squeeze(-1))
        return GATE_FLOOR + (1 - 2 * GATE_FLOOR) * torch.sigmoid(base + delta)


def mixture_nll(weight: torch.Tensor, neural: torch.Tensor,
                count: torch.Tensor) -> torch.Tensor:
    if not (weight.shape == neural.shape == count.shape):
        raise ValueError("Mixture target arrays have mismatched shapes")
    if (weight <= 0).any() or (weight >= 1).any() or (neural <= 0).any() or (count < 0).any():
        raise ValueError("Invalid target probabilities or mixture weights")
    return -torch.logaddexp(
        torch.log1p(-weight.double()) + neural.double().log(),
        weight.double().log() + count.double().log(),
    )


def fit_direction(features: torch.Tensor, old_weight: torch.Tensor,
                  neural: torch.Tensor, count: torch.Tensor,
                  fit_mask: torch.Tensor, hold_mask: torch.Tensor,
                  device: torch.device) -> dict:
    fit_x = features[fit_mask]
    hold_x = features[hold_mask]
    mean = fit_x.mean(0)
    std = fit_x.std(0).clamp_min(1e-4)
    fit_x = ((fit_x - mean) / std).clamp(-10, 10).to(device)
    hold_x = ((hold_x - mean) / std).clamp(-10, 10).to(device)
    fit_w = old_weight[fit_mask].to(device)
    hold_w = old_weight[hold_mask].to(device)
    fit_neural = neural[fit_mask].to(device)
    fit_count = count[fit_mask].to(device)
    hold_neural = neural[hold_mask].to(device)
    hold_count = count[hold_mask].to(device)

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    gate = ResidualGate(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    before = float(mixture_nll(hold_w, hold_neural, hold_count).sum())
    initial = float(mixture_nll(gate(hold_x, hold_w), hold_neural, hold_count).sum())
    if abs(initial - before) > 0.003:
        raise ValueError("Zero-start residual gate does not reproduce Stage143")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    training_losses = []
    gate.train()
    for _ in range(EPOCHS):
        order = torch.randperm(fit_x.shape[0], generator=generator).to(device)
        for indices in order.split(BATCH_SIZE):
            predicted = gate(fit_x[indices], fit_w[indices])
            loss = mixture_nll(predicted, fit_neural[indices], fit_count[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            training_losses.append(float(mixture_nll(
                gate(fit_x, fit_w), fit_neural, fit_count).mean()))
    gate.eval()
    with torch.inference_mode():
        after = float(mixture_nll(gate(hold_x, hold_w),
                                  hold_neural, hold_count).sum())
        mean_hold_weight = float(gate(hold_x, hold_w).mean())
    return dict(fit_targets=int(fit_mask.sum()), heldout_targets=int(hold_mask.sum()),
                baseline_heldout_nll_nats=before, initial_heldout_nll_nats=initial,
                fitted_heldout_nll_nats=after,
                improvement_nats=before - after,
                mean_fitted_heldout_count_weight=mean_hold_weight,
                train_mean_nll_nats_by_epoch=training_losses)


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
        raise ValueError("Unexpected checkpoint ancestry")
    model, implementation_sha = make_model(checkpoint["implementation"],
                                            checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    validation, byte_count = load_data()["validation"]
    if byte_count != EXPECTED_BYTES:
        raise ValueError("Validation byte coverage changed")

    current_targets: torch.Tensor | None = None
    hidden_capture: list[torch.Tensor] = []
    count_capture: list[tuple[torch.Tensor, torch.Tensor]] = []
    target_capture: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    original_features = model.neural.features
    original_collect = model.ngram.collect
    original_add = model.ngram.add_collected

    def capture_features(ids):
        hidden = original_features(ids)
        hidden_capture.append(hidden)
        return hidden

    def capture_collect(ids):
        collection = original_collect(ids)
        count_capture.append((collection[2], collection[3]))
        return collection

    def capture_add(result, weight, terms, suffix):
        if current_targets is None:
            raise ValueError("Missing current target batch")
        target_ids = current_targets.clamp_min(0).unsqueeze(-1)
        top_two = result.topk(2, dim=-1).values.clone()
        before = result.gather(-1, target_ids).squeeze(-1).clone()
        returned = original_add(result, weight, terms, suffix)
        after = returned.gather(-1, target_ids).squeeze(-1).clone()
        target_capture.append((before, after, weight.clone(), top_two))
        return returned

    model.neural.features = capture_features
    model.ngram.collect = capture_collect
    model.ngram.add_collected = capture_add
    features_list = []
    neural_list = []
    count_list = []
    weight_list = []
    window_list = []
    total_nll = 0.0
    rows_seen = 0
    with torch.inference_mode():
        for x, y in windows(validation, batch_size=32):
            current_targets = y
            hidden_capture.clear()
            count_capture.clear()
            target_capture.clear()
            logp = model.predict_log_probs(x)
            if not (len(hidden_capture) == len(count_capture) == len(target_capture) == 1):
                raise ValueError("Unexpected number of causal feature/lookup calls")
            valid = y != -100
            hidden = hidden_capture[0][valid]
            backoff, max_mass = count_capture[0]
            before, after, weight, top_two = target_capture[0]
            neural, count = recover_target_probabilities(
                before[valid], after[valid], weight[valid])
            observed = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid]
            if (after[valid].float() - observed.exp()).abs().max() > 2e-6:
                raise ValueError("Target probability does not match frozen predictor")
            total_nll -= float(observed.double().sum())
            position = torch.arange(x.shape[1]).expand(x.shape[0], -1)
            top_two = top_two[valid].double() / (1 - weight[valid].double()).unsqueeze(-1)
            if (top_two <= 0).any():
                raise ValueError("Invalid neural confidence features")
            base_weight = weight[valid].double()
            base_logit = ((base_weight - GATE_FLOOR) / (1 - 2 * GATE_FLOOR))
            base_logit = torch.logit(base_logit.clamp(1e-6, 1 - 1e-6))
            extras = torch.stack((
                base_logit, top_two[:, 0].log(),
                (top_two[:, 0] / top_two[:, 1]).log(),
                position[valid].double() / 255,
                backoff[valid].double(), max_mass[valid].double(),
            ), dim=-1).float()
            features_list.append(torch.cat((hidden.float(), extras), dim=-1).cpu())
            neural_list.append(neural.cpu())
            count_list.append(count.cpu())
            weight_list.append(weight[valid].cpu())
            window_ids = torch.arange(rows_seen, rows_seen + x.shape[0])[:, None]
            window_list.append(window_ids.expand_as(x)[valid].cpu())
            rows_seen += x.shape[0]

    features = torch.cat(features_list)
    neural = torch.cat(neural_list)
    count = torch.cat(count_list)
    old_weight = torch.cat(weight_list)
    window_ids = torch.cat(window_list)
    if len(features) != EXPECTED_TARGETS or rows_seen != math.ceil(EXPECTED_TARGETS / 256):
        raise ValueError("Incomplete independent-window coverage")
    scale = math.log(2) * byte_count
    reference_bpb = total_nll / scale
    reconstructed_bpb = float(mixture_nll(old_weight, neural, count).sum()) / scale
    if abs(reference_bpb - REFERENCE_BPB) > 2e-5 or abs(reconstructed_bpb - reference_bpb) > 2e-5:
        raise ValueError("Unchanged Stage143 complete-validation score mismatch")
    first = window_ids < rows_seen // 2
    second = ~first
    fit_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directions = {
        "first_to_second": fit_direction(features, old_weight, neural, count,
                                         first, second, fit_device),
        "second_to_first": fit_direction(features, old_weight, neural, count,
                                         second, first, fit_device),
    }
    combined_nll = sum(row["fitted_heldout_nll_nats"] for row in directions.values())
    combined_bpb = combined_nll / scale
    result = dict(
        protocol=PROTOCOL, purpose="validation_fitted_out_of_half_gate_diagnostic_not_deployable",
        split="validation", precision=precision, feature_device=str(device),
        fitting_device=str(fit_device), threads=args.threads,
        checkpoint_sha256=CHECKPOINT_SHA256,
        implementation_sha256=implementation_sha,
        stage175_source_sha256=sha(ROOT / "scripts/diagnose_stage175_current_expert_oracle.py"),
        source_sha256=sha(Path(__file__)),
        graph_sha256=sha(ROOT / "inference_assets/stage143-stage92-features.onnx"),
        targets=len(features), utf8_bytes=byte_count, independent_windows=rows_seen,
        feature_count=features.shape[1], split_window_index=rows_seen // 2,
        seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, hidden_width=HIDDEN_WIDTH,
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        unchanged_stage143_bpb=reference_bpb,
        reconstructed_stage143_bpb=reconstructed_bpb,
        directions=directions,
        combined_out_of_half_bpb=combined_bpb,
        improvement_vs_stage143_bpb=reference_bpb - combined_bpb,
        advancement_gate_bpb=1.35,
        advancement_gate_passed=(combined_bpb <= 1.35 and all(
            row["improvement_nats"] > 0 for row in directions.values())),
        no_test_scoring=True,
        warning="Coefficients fitted to validation answers are diagnostic only and cannot be deployed or submitted.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
