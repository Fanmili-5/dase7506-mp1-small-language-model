"""Export and independently qualify the Stage102 train-learned gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from common import PROTOCOL, make_model, setup, sha
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.fit_stage100_train_gate import (
    EXTENDED_SHA, FEATURE_COLUMNS, FEATURE_NAMES, confidence_features5)
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload


TRAIN_GATE_SHA = "8608489f8c2ea6df6909ba8ff52557c024ba9dbd3c8b7032246d2579b435e991"
STAGE102_SHA = "fcae8215e5b83ab66059fdafd0b4319defd240fa1bcd0ca93764fe34df88704f"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
EXPECTED_BPB = 1.399686163160504
SELECTED_NAMES = (
    "neural_max_logp", "neural_margin", "highest_backoff", "highest_max_mass")
INFERENCE_FILES = (
    "student_stage103_gated.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "common.py", "evaluate.py", "data/tokenizer.json", "requirements.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != EXTENDED_SHA
            or sha(args.train_gate) != TRAIN_GATE_SHA
            or sha(args.selection) != STAGE102_SHA
            or sha(args.baseline) != BASELINE_SHA):
        raise ValueError("Unexpected pinned candidate ancestry")
    train_gate = json.loads(args.train_gate.read_text(encoding="utf-8-sig"))
    selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
    best = selection["best"]
    if (train_gate.get("no_validation_or_test_fitting") is not True
            or train_gate.get("feature_names") != list(FEATURE_NAMES)
            or best.get("variant") != "neural_backoff_mass"
            or best.get("active_features") != list(SELECTED_NAMES)
            or best.get("slope_scale") != .5
            or abs(best.get("bpb") - EXPECTED_BPB) > 1e-10):
        raise ValueError("Unexpected train gate or selected setting")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or count_payload.get("implementation") != "student_ngram"
            or count_payload["config"].get("max_order") != 6):
        raise ValueError("Unexpected expert checkpoint payload")
    config = dict(count_payload["config"], kind="gated_hybrid",
                  neural_config=neural_payload["config"],
                  feature_names=list(SELECTED_NAMES),
                  vocabulary_temperature=1.125,
                  train_unigram_prior_weight=.0625,
                  copy_gate_shift=.25,
                  anchor_weight=.0625, slope_scale=.5)
    device, _ = setup("cpu", "fp32", 4)
    candidate, implementation_sha = make_model("student_stage103_gated", config, device)
    candidate.neural.load_state_dict(neural_payload["model"], strict=True)
    candidate.ngram.load_state_dict(count_payload["model"], strict=True)
    log_prior, train_unigram_tokens = train_log_prior()
    with torch.no_grad():
        candidate.log_prior.copy_(log_prior)
        indices = [FEATURE_NAMES.index(name) for name in SELECTED_NAMES]
        candidate.gate_mean.copy_(torch.tensor(
            [train_gate["feature_mean"][i] for i in indices], dtype=torch.float64))
        candidate.gate_std.copy_(torch.tensor(
            [train_gate["feature_std"][i] for i in indices], dtype=torch.float64))
        candidate.gate_coeff.copy_(torch.tensor(
            [train_gate["coefficients"][name] for name in SELECTED_NAMES],
            dtype=torch.float64))
    candidate.eval()
    reference_counts, _ = make_model("student_ngram", count_payload["config"], device)
    reference_counts.load_state_dict(count_payload["model"], strict=True)
    reference_counts.eval()
    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        neural_logp = distilled_log_probs(candidate.neural, smoke_ids, log_prior)
        count_logp = reference_counts.predict_log_probs(smoke_ids)
        all_features = confidence_features5(
            neural_logp, count_logp, reference_counts, smoke_ids)
        selected = all_features[:, [FEATURE_COLUMNS[i] for i in indices]].double()
        slope = ((selected - candidate.gate_mean) / candidate.gate_std)
        slope = (slope * candidate.gate_coeff).sum(-1)
        anchor = math.log(.0625 / .9375)
        weight = torch.sigmoid(anchor + .5 * slope)
        weight = 1e-4 + (1 - 2e-4) * weight
        reference = torch.logaddexp(
            neural_logp + torch.log1p(-weight.float().reshape_as(smoke_ids)).unsqueeze(-1),
            count_logp + weight.float().reshape_as(smoke_ids).log().unsqueeze(-1))
        predicted = candidate.predict_log_probs(smoke_ids)
    max_probability_error = float((reference.exp() - predicted.exp()).abs().max())
    max_logp_error = float((reference - predicted).abs().max())
    max_normalization_error = float(predicted.logsumexp(-1).abs().max())
    if (not torch.isfinite(predicted).all()
            or max_probability_error > 3e-6 or max_logp_error > 3e-5
            or max_normalization_error > 1e-5):
        raise ValueError("Stage103 smoke equivalence or normalization failed: "
                         f"{max_probability_error}, {max_logp_error}, "
                         f"{max_normalization_error}")

    args.run_dir.mkdir(parents=True)
    checkpoint = args.run_dir / "stage103-gated-order6.pt"
    payload = checkpoint_payload(
        candidate, "student_stage103_gated", config,
        int(neural_payload.get("seed", 92017)),
        int(neural_payload.get("train_tokens", 0)))
    payload["train_gate_ancestry"] = dict(
        neural_sha256=NEURAL_SHA, count_sha256=EXTENDED_SHA,
        train_gate_sha256=TRAIN_GATE_SHA,
        validation_setting_sha256=STAGE102_SHA,
        train_gate_fit_targets=train_gate["gate_fit_targets"],
        train_gate_selection_targets=train_gate["gate_selection_targets"],
        selected_feature_names=list(SELECTED_NAMES),
        slope_scale=.5, anchor_weight=.0625,
        train_unigram_tokens=train_unigram_tokens,
        no_validation_or_test_gradient_fitting=True,
        no_test_scoring=True)
    atomic_torch_save(payload, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resources_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if abs(validation["bpb"] - EXPECTED_BPB) > 2e-6:
        raise ValueError("Stage103 full validation disagrees with Stage102")
    subprocess.run([
        sys.executable, "scripts/benchmark_cpu.py",
        "--baseline", str(args.baseline.resolve()),
        "--candidate", str(checkpoint.resolve()), "--repeats", "3",
        "--threads", "4", "--output", str(resources_path.resolve()),
    ], cwd=ROOT, check=True)
    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    assets = checkpoint.stat().st_size + sum(
        (ROOT / name).stat().st_size for name in INFERENCE_FILES)
    final = dict(
        protocol=PROTOCOL, status="stage103_export_resource_audited",
        checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        neural_sha256=NEURAL_SHA, counts_sha256=EXTENDED_SHA,
        validation_setting_sha256=STAGE102_SHA,
        validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        smoke_max_probability_error=max_probability_error,
        smoke_max_logp_error=max_logp_error,
        smoke_max_normalization_error=max_normalization_error,
        inference_files=INFERENCE_FILES,
        source_hashes={name: sha(ROOT / name) for name in INFERENCE_FILES},
        split="validation", precision="fp32", no_test_scoring=True)
    final["qualified"] = all((
        final["within_five_x_time_limit"],
        final["within_four_gib_peak_rss_limit"],
        final["within_64mib_asset_limit"],
    ))
    atomic_json_dump(final, args.run_dir / "final.json")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
