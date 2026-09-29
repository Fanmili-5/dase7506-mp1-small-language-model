"""Validation-crossfit token-wise count gate diagnostic; never deploy its offsets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows


CHECKPOINT_SHA256 = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
REFERENCE_BPB = 1.399686162042141
EXPECTED_TARGETS = 376_599
EXPECTED_BYTES = 1_148_007
EXPECTED_WINDOWS = 1_472
SPLIT_WINDOW = 736
VOCAB = 2_048
GATE_FLOOR = 1e-4
REGULARIZER = 10.0
OFFSET_CAP = 2.0


def validation_only() -> tuple[torch.Tensor, int, dict[str, str]]:
    """Verify/read only tokenizer and validation; never open supplied test text."""
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Protocol changed")
    hashes = manifest["sha256"]
    tokenizer_path = ROOT / "data/tokenizer.json"
    validation_path = ROOT / "data/wikitext_validation.txt"
    if (sha(tokenizer_path) != hashes["tokenizer.json"]
            or sha(validation_path) != hashes["wikitext_validation.txt"]):
        raise ValueError("Tokenizer or validation changed")
    raw = validation_path.read_bytes()
    if len(raw) != EXPECTED_BYTES:
        raise ValueError("Validation byte coverage changed")
    ids = Tokenizer.from_file(str(tokenizer_path)).encode(raw.decode("utf-8")).ids
    return torch.tensor(ids, dtype=torch.long), len(raw), {
        "tokenizer_sha256": hashes["tokenizer.json"],
        "validation_sha256": hashes["wikitext_validation.txt"],
    }


def gate_derivative(weight: torch.Tensor) -> torch.Tensor:
    scaled = ((weight - GATE_FLOOR) / (1 - 2 * GATE_FLOOR)).clamp(1e-6, 1 - 1e-6)
    return (1 - 2 * GATE_FLOOR) * scaled * (1 - scaled)


def shifted_weight(weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    scaled = ((weight - GATE_FLOOR) / (1 - 2 * GATE_FLOOR)).clamp(1e-6, 1 - 1e-6)
    return GATE_FLOOR + (1 - 2 * GATE_FLOOR) * torch.sigmoid(
        torch.logit(scaled)[:, None] + offsets[None, :])


def fit_one_step(gradient: torch.Tensor, fisher: torch.Tensor) -> torch.Tensor:
    if (gradient.shape != (VOCAB,) or fisher.shape != (VOCAB,)
            or not torch.isfinite(gradient).all() or not torch.isfinite(fisher).all()
            or (fisher < 0).any()):
        raise ValueError("Invalid token-wise fit statistics")
    return (gradient / (fisher + REGULARIZER)).clamp(-OFFSET_CAP, OFFSET_CAP).float()


def score_gradient_rows(neural: torch.Tensor, count: torch.Tensor,
                        weight: torch.Tensor, mixed: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """Exact first derivative of normalized target log-probability at b=0."""
    derivative = gate_derivative(weight)[:, None] * (count - neural)
    target_p = mixed.gather(1, target[:, None]).squeeze(1)
    target_d = derivative.gather(1, target[:, None]).squeeze(1)
    score_gradient = -derivative
    score_gradient.scatter_add_(1, target[:, None], (target_d / target_p)[:, None])
    return score_gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("Use a new output file and positive thread count")
    if sha(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Protected Stage143 checkpoint changed")
    device, precision = setup("cpu", "fp32", args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("implementation") != "student_stage143_openvino_singlepass"):
        raise ValueError("Unexpected frozen predictor")
    model, implementation_sha = make_model(checkpoint["implementation"],
                                            checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    validation, byte_count, data_hashes = validation_only()

    original_add = model.ngram.add_collected
    captures: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_add(result, weight, terms, suffix):
        neural = result / (1 - weight).unsqueeze(-1)
        count = original_add(torch.zeros_like(result), torch.ones_like(weight), terms, suffix)
        mixed = original_add(result, weight, terms, suffix)
        captures.append((neural, count, weight.clone(), mixed))
        return mixed

    model.ngram.add_collected = capture_add

    def batches():
        rows_seen = 0
        for x, y in windows(validation, batch_size=32):
            if rows_seen < SPLIT_WINDOW < rows_seen + x.shape[0]:
                raise ValueError("Validation half split bisects a batch")
            half = 0 if rows_seen < SPLIT_WINDOW else 1
            captures.clear()
            with torch.inference_mode():
                logp = model.predict_log_probs(x)
            if len(captures) != 1:
                raise ValueError("Expected exactly one count addition per batch")
            neural, count, weight, mixed = captures[0]
            valid = y != -100
            targets = y[valid]
            neural = neural[valid]
            count = count[valid]
            weight = weight[valid]
            mixed = mixed[valid]
            observed = logp[valid].gather(1, targets[:, None]).squeeze(1)
            if (neural.shape[1] != VOCAB or not torch.isfinite(neural).all()
                    or not torch.isfinite(count).all() or (neural < 0).any()
                    or (count < -1e-6).any() or (mixed <= 0).any()
                    or ((weight <= 0) | (weight >= 1)).any()):
                raise ValueError("Invalid complete expert distributions")
            max_mass_error = max(
                float((neural.sum(1) - 1).abs().max()),
                float((count.sum(1) - 1).abs().max()),
                float((mixed.sum(1) - 1).abs().max()),
            )
            if max_mass_error > 3e-4:
                raise ValueError(f"Expert probability mass changed: {max_mass_error}")
            zero = (1 - weight[:, None]) * neural + weight[:, None] * count
            max_zero_error = float((zero - mixed).abs().max())
            if max_zero_error > 2e-6 or float((mixed.gather(1, targets[:, None]).squeeze(1).log() - observed).abs().max()) > 2e-5:
                raise ValueError("Zero-offset mixture does not reproduce Stage143")
            yield half, targets, neural, count.clamp_min(0), weight, mixed, observed, max_zero_error, max_mass_error
            rows_seen += x.shape[0]
        if rows_seen != EXPECTED_WINDOWS:
            raise ValueError("Incomplete independent-window coverage")

    stats = [{"gradient": torch.zeros(VOCAB, dtype=torch.float64),
              "fisher": torch.zeros(VOCAB, dtype=torch.float64),
              "baseline_nll": 0.0, "targets": 0} for _ in range(2)]
    maximum_zero_error = 0.0
    maximum_mass_error = 0.0
    for half, target, neural, count, weight, mixed, observed, zero_error, mass_error in batches():
        row = stats[half]
        score_gradient = score_gradient_rows(neural, count, weight, mixed, target)
        row["gradient"] += score_gradient.double().sum(0)
        row["fisher"] += score_gradient.double().square().sum(0)
        row["baseline_nll"] -= float(observed.double().sum())
        row["targets"] += len(target)
        maximum_zero_error = max(maximum_zero_error, zero_error)
        maximum_mass_error = max(maximum_mass_error, mass_error)

    if sum(row["targets"] for row in stats) != EXPECTED_TARGETS:
        raise ValueError("Incomplete target coverage")
    offsets = [fit_one_step(row["gradient"], row["fisher"]) for row in stats]
    heldout_nll = [0.0, 0.0]
    repeated_baseline_nll = [0.0, 0.0]
    repeated_targets = [0, 0]
    for half, target, neural, count, weight, mixed, observed, _, _ in batches():
        candidate_weight = shifted_weight(weight, offsets[1 - half])
        raw = (1 - candidate_weight) * neural + candidate_weight * count
        normalizer = raw.double().sum(1)
        target_probability = raw.gather(1, target[:, None]).squeeze(1).double() / normalizer
        if (not torch.isfinite(target_probability).all()
                or (target_probability <= 0).any() or (normalizer <= 0).any()):
            raise ValueError("Token-wise distribution is invalid")
        heldout_nll[half] -= float(target_probability.log().sum())
        repeated_baseline_nll[half] -= float(observed.double().sum())
        repeated_targets[half] += len(target)

    baseline_nll = sum(row["baseline_nll"] for row in stats)
    if (any(abs(row["baseline_nll"] - repeated_baseline_nll[i]) > 0.01
            or row["targets"] != repeated_targets[i] for i, row in enumerate(stats))
            or abs(baseline_nll / (math.log(2) * byte_count) - REFERENCE_BPB) > 2e-5):
        raise ValueError("Frozen baseline identity or repeat coverage changed")
    half_records = []
    for half in range(2):
        o = offsets[1 - half]
        half_records.append({
            "fitted_on_half": 1 - half,
            "evaluated_on_half": half,
            "heldout_targets": stats[half]["targets"],
            "baseline_heldout_nll_nats": stats[half]["baseline_nll"],
            "tokenwise_heldout_nll_nats": heldout_nll[half],
            "heldout_nll_gain_nats": stats[half]["baseline_nll"] - heldout_nll[half],
            "offset_min": float(o.min()), "offset_max": float(o.max()),
            "offset_mean_abs": float(o.abs().mean()),
            "offset_clipped_count": int((o.abs() >= OFFSET_CAP - 1e-6).sum()),
        })
    combined_bpb = sum(heldout_nll) / (math.log(2) * byte_count)
    result = {
        "protocol": PROTOCOL,
        "purpose": "validation_fitted_tokenwise_gate_crossfit_diagnostic_not_deployable",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": implementation_sha,
        "graph_sha256": sha(ROOT / "inference_assets/stage143-stage92-features.onnx"),
        "source_sha256": sha(Path(__file__)),
        **data_hashes,
        "split": "validation", "device": str(device), "precision": precision,
        "threads": args.threads, "targets": EXPECTED_TARGETS, "utf8_bytes": byte_count,
        "independent_windows": EXPECTED_WINDOWS, "split_window_index": SPLIT_WINDOW,
        "regularizer": REGULARIZER, "offset_cap": OFFSET_CAP,
        "fitting_rule": "one_regularized_empirical_diagonal_fisher_step_per_token",
        "stage143_bpb": baseline_nll / (math.log(2) * byte_count),
        "combined_out_of_half_bpb": combined_bpb,
        "gain_bpb": (baseline_nll - sum(heldout_nll)) / (math.log(2) * byte_count),
        "halves": half_records,
        "maximum_zero_offset_probability_error": maximum_zero_error,
        "maximum_expert_mass_error": maximum_mass_error,
        "advancement_gate_bpb": 1.365,
        "advancement_gate_passed": combined_bpb <= 1.365 and all(
            row["heldout_nll_gain_nats"] > 0 for row in half_records),
        "no_test_scoring": True,
        "warning": "Offsets fitted to validation labels are diagnostic only; never deploy or test them.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
