"""Materialize Stage80 temperature into an untied output head and requalify."""
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


STAGE80_SHA = "23e458f0870d25ed0487d9c72082cc13045e0d70ce2621eacb35f4fd789d9373"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
INFERENCE_FILES = (
    "student_ngram_hybrid_conv_calibrated_untied_collapsed.py",
    "student_ngram_hybrid_conv_bias_collapsed.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "student_structured.py", "student.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage80", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if sha(args.stage80) != STAGE80_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Unexpected Stage80 or baseline checkpoint")
    payload = torch.load(args.stage80, map_location="cpu", weights_only=True)
    calibration = payload.get("fixed_calibration", {})
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation")
            != "student_ngram_hybrid_conv_calibrated_collapsed"
            or not calibration.get("no_new_gradient_targets")):
        raise ValueError("Expected the fixed Stage80 calibrated checkpoint")
    temperature = float(calibration["vocabulary_temperature"])
    config = dict(payload["config"], kind="hybrid_calibrated_untied_conv",
                  materialized_vocabulary_temperature=temperature)
    device, _ = setup("cpu", "fp32", 4)
    reference, _ = make_model(payload["implementation"], payload["config"], device)
    reference.load_state_dict(payload["model"]); reference.eval()
    candidate, _ = make_model(
        "student_ngram_hybrid_conv_calibrated_untied_collapsed", config, device)
    candidate.load_state_dict(payload["model"], strict=True)
    with torch.no_grad():
        candidate.neural.head.weight.div_(temperature)
    candidate.eval()

    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        expected = reference.predict_log_probs(smoke_ids)
        actual = candidate.predict_log_probs(smoke_ids)
    max_probability_error = float((actual.exp() - expected.exp()).abs().max())
    max_logp_error = float((actual - expected).abs().max())
    if (not torch.isfinite(actual).all() or max_probability_error > 3e-6
            or max_logp_error > 3e-5):
        raise ValueError(
            f"Untied output materialization mismatch: {max_probability_error}, "
            f"{max_logp_error}")

    args.run_dir.mkdir(parents=True)
    exported = dict(
        payload,
        implementation="student_ngram_hybrid_conv_calibrated_untied_collapsed",
        config=config,
        model=candidate.state_dict(),
    )
    exported["inference_optimization"] = dict(
        source="Stage80 exact calibrated checkpoint",
        source_checkpoint_sha256=STAGE80_SHA,
        method="materialize_temperature_into_untied_output_matrix",
        vocabulary_temperature=temperature,
        no_new_gradient_targets=True,
        max_smoke_probability_error=max_probability_error,
        max_smoke_logp_error=max_logp_error,
    )
    checkpoint = args.run_dir / "untied-calibrated-collapsed-hybrid.pt"
    atomic_torch_save(exported, checkpoint)

    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resources_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    selected_bpb = float(calibration["selected_validation_bpb"])
    if abs(validation["bpb"] - selected_bpb) > 2e-6:
        raise ValueError("Stage82 CPU score disagrees with the frozen calibration")
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
        protocol=PROTOCOL, status="equivalent_untied_resource_audited",
        checkpoint_sha256=sha(checkpoint), source_checkpoint_sha256=STAGE80_SHA,
        baseline_checkpoint_sha256=BASELINE_SHA,
        validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        inference_optimization=exported["inference_optimization"],
        fixed_calibration=calibration, split="validation", precision="fp32",
        no_test_scoring=True,
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
