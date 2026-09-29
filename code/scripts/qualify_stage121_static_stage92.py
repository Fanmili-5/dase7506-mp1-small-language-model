"""Export the fixed Stage92/Stage94 order-five mixture through the fast Stage85 graph."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, make_model, setup, sha
from scripts.analyze_stage98_distilled_gate import (
    GATE_SHIFT, NEURAL_SHA, PRIOR_WEIGHT, TEMPERATURE, distilled_log_probs)
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload


BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
EXPECTED_BPB = 1.4017076622984082
COUNT_WEIGHT = .0625
IMPLEMENTATION = "student_ngram_hybrid_conv_fused_residual_norm"
INFERENCE_FILES = (
    "student_ngram_hybrid_conv_fused_residual_norm.py",
    "student_ngram_hybrid_conv_fused_copy_collapsed.py",
    "student_ngram_hybrid_conv_bias_collapsed.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "student_structured.py", "student.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage121 output directory")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != BASE_SHA
            or sha(args.baseline) != BASELINE_SHA):
        raise ValueError("Unexpected frozen Stage92, order-five, or baseline asset")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"
            or count_payload["config"].get("max_order") != 5):
        raise ValueError("Unexpected expert checkpoint format")
    config = dict(count_payload["config"], kind="hybrid",
                  neural_config=neural_payload["config"],
                  mixture_weight=COUNT_WEIGHT)
    device, _ = setup("cpu", "fp32", 4)
    candidate, _ = make_model(IMPLEMENTATION, config, device)
    candidate.neural.load_state_dict(neural_payload["model"], strict=True)
    candidate.ngram.load_state_dict(count_payload["model"], strict=True)
    reference_neural, _ = make_model(
        neural_payload["implementation"], neural_payload["config"], device)
    reference_counts, _ = make_model("student_ngram", count_payload["config"], device)
    reference_neural.load_state_dict(neural_payload["model"], strict=True)
    reference_counts.load_state_dict(count_payload["model"], strict=True)
    reference_neural.eval(); reference_counts.eval()
    log_prior, train_unigram_tokens = train_log_prior()
    with torch.no_grad():
        # Exact Stage94 calibration is folded into existing affine parameters.
        candidate.neural.norm.weight.div_(TEMPERATURE)
        if getattr(candidate.neural.norm, "bias", None) is not None:
            candidate.neural.norm.bias.div_(TEMPERATURE)
        candidate.neural.output_bias.div_(TEMPERATURE)
        candidate.neural.output_bias.add_(PRIOR_WEIGHT * log_prior)
        candidate.neural.copy_query.weight.mul_(TEMPERATURE)
        candidate.neural.copy_key.weight.mul_(TEMPERATURE)
        candidate.neural.copy_gate.weight.mul_(TEMPERATURE)
        candidate.neural.copy_gate.bias.add_(GATE_SHIFT)
    candidate.eval()
    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        neural_logp = distilled_log_probs(reference_neural, smoke_ids, log_prior)
        count_logp = reference_counts.predict_log_probs(smoke_ids)
        expected = torch.logaddexp(
            neural_logp + math.log1p(-COUNT_WEIGHT),
            count_logp + math.log(COUNT_WEIGHT))
        actual = candidate.predict_log_probs(smoke_ids)
    smoke = dict(
        max_probability_error=float((expected.exp() - actual.exp()).abs().max()),
        max_logp_error=float((expected - actual).abs().max()),
        max_normalization_error=float(actual.logsumexp(-1).abs().max()))
    if (not torch.isfinite(actual).all()
            or smoke["max_probability_error"] > 3e-6
            or smoke["max_logp_error"] > 3e-4
            or smoke["max_normalization_error"] > 1e-5):
        raise ValueError(f"Stage121 folded smoke mismatch: {smoke}")
    args.run_dir.mkdir(parents=True)
    checkpoint = args.run_dir / "stage121-static-stage92-order5.pt"
    payload = checkpoint_payload(
        candidate, IMPLEMENTATION, config,
        int(neural_payload.get("seed", 92017)),
        int(neural_payload.get("train_tokens", 0)))
    payload["static_ancestry"] = dict(
        neural_sha256=NEURAL_SHA, counts_sha256=BASE_SHA,
        train_unigram_tokens=train_unigram_tokens,
        temperature=TEMPERATURE, prior_weight=PRIOR_WEIGHT,
        copy_gate_shift=GATE_SHIFT, count_weight=COUNT_WEIGHT,
        no_new_training_targets=True, no_test_scoring=True, smoke=smoke)
    atomic_torch_save(payload, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resource_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if abs(validation["bpb"] - EXPECTED_BPB) > 2e-6:
        raise ValueError("Stage121 exact full-validation score disagrees with Stage114")
    subprocess.run([
        sys.executable, "scripts/benchmark_cpu.py", "--baseline",
        str(args.baseline.resolve()), "--candidate", str(checkpoint.resolve()),
        "--repeats", "3", "--threads", "4", "--output",
        str(resource_path.resolve()),
    ], cwd=ROOT, check=True)
    resources = json.loads(resource_path.read_text(encoding="utf-8"))
    assets = checkpoint.stat().st_size + sum(
        (ROOT / file).stat().st_size for file in INFERENCE_FILES)
    final = dict(
        protocol=PROTOCOL, status="stage121_static_order5_resource_audited",
        checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        static_ancestry=payload["static_ancestry"],
        inference_files=INFERENCE_FILES,
        source_hashes={file: sha(ROOT / file) for file in INFERENCE_FILES},
        split="validation", precision="fp32", no_test_scoring=True)
    final["qualified"] = all((final["within_five_x_time_limit"],
                               final["within_four_gib_peak_rss_limit"],
                               final["within_64mib_asset_limit"]))
    atomic_json_dump(final, args.run_dir / "final.json")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
