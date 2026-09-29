"""Remove Stage84's redundant dense normalization division and requalify."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import PROTOCOL, make_model, setup, sha
from train_experiment import atomic_json_dump, atomic_torch_save


STAGE84_SHA = "d3a21c8e94f345bc95ddc5b2b7e7e02a4dcfefae8afaafdac3c234b7bcde7542"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
INFERENCE_FILES = (
    "student_ngram_hybrid_conv_fused_residual_norm.py",
    "student_ngram_hybrid_conv_fused_copy_collapsed.py",
    "student_ngram_hybrid_conv_bias_collapsed.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "student_structured.py", "student.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage84", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if sha(args.stage84) != STAGE84_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Unexpected Stage84 or baseline checkpoint")

    payload = torch.load(args.stage84, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation")
            != "student_ngram_hybrid_conv_fused_copy_collapsed"
            or not payload.get("fused_copy_optimization", {}).get(
                "no_dense_copy_probability_temporary")):
        raise ValueError("Expected the fused-copy Stage84 checkpoint")

    device, _ = setup("cpu", "fp32", 4)
    reference, _ = make_model(payload["implementation"], payload["config"], device)
    reference.load_state_dict(payload["model"]); reference.eval()
    candidate, _ = make_model(
        "student_ngram_hybrid_conv_fused_residual_norm",
        payload["config"], device)
    candidate.load_state_dict(payload["model"], strict=True); candidate.eval()

    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        expected = reference.predict_log_probs(smoke_ids)
        actual = candidate.predict_log_probs(smoke_ids)
    max_probability_error = float((actual.exp() - expected.exp()).abs().max())
    max_logp_error = float((actual - expected).abs().max())
    max_normalization_error = float(actual.logsumexp(-1).abs().max())
    if (not torch.isfinite(actual).all() or max_probability_error > 3e-6
            or max_logp_error > 3e-5 or max_normalization_error > 1e-5):
        raise ValueError(
            "Residual normalization mismatch: "
            f"{max_probability_error}, {max_logp_error}, "
            f"{max_normalization_error}")

    args.run_dir.mkdir(parents=True)
    exported = dict(
        payload,
        implementation="student_ngram_hybrid_conv_fused_residual_norm",
        model=candidate.state_dict(),
    )
    exported["normalization_optimization"] = dict(
        source="Stage84 exact fused-copy checkpoint",
        source_checkpoint_sha256=STAGE84_SHA,
        method="direct_log_of_already_normalized_convex_mixture",
        no_dense_output_division=True,
        no_new_gradient_targets=True,
        max_smoke_probability_error=max_probability_error,
        max_smoke_logp_error=max_logp_error,
        max_smoke_normalization_error=max_normalization_error,
    )
    checkpoint = args.run_dir / "direct-log-calibrated-collapsed-hybrid.pt"
    atomic_torch_save(exported, checkpoint)

    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resources_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    selected_bpb = float(payload["fixed_calibration"]["selected_validation_bpb"])
    if abs(validation["bpb"] - selected_bpb) > 2e-6:
        raise ValueError("Stage85 CPU score disagrees with the frozen calibration")
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
        protocol=PROTOCOL, status="residual_norm_resource_audited",
        checkpoint_sha256=sha(checkpoint), source_checkpoint_sha256=STAGE84_SHA,
        baseline_checkpoint_sha256=BASELINE_SHA,
        validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        normalization_optimization=exported["normalization_optimization"],
        fused_copy_optimization=payload["fused_copy_optimization"],
        fixed_calibration=payload["fixed_calibration"], split="validation",
        precision="fp32", no_test_scoring=True,
        source_hashes={name: sha(ROOT / name) for name in INFERENCE_FILES},
    )
    final["qualified"] = all((
        final["within_five_x_time_limit"],
        final["within_four_gib_peak_rss_limit"],
        final["within_64mib_asset_limit"],
    ))
    atomic_json_dump(final, args.run_dir / "final.json")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
