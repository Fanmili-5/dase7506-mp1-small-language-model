"""Verify fused Stage104 against Stage103 and run all resource gates."""
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


STAGE103_SHA = "a58011ed846f007b053c7a430837201223bec2d5611480e5a6cf3ba8f97ba7c2"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
EXPECTED_BPB = 1.399686163160504
INFERENCE_FILES = (
    "student_stage104_gated_fast.py", "student_stage103_gated.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram_collapsed.py",
    "student_ngram_fast.py", "student_ngram.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage103", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if sha(args.stage103) != STAGE103_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Unexpected Stage103 or baseline checkpoint")
    payload = torch.load(args.stage103, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_stage103_gated"):
        raise ValueError("Unexpected Stage103 payload")
    device, _ = setup("cpu", "fp32", 4)
    reference, _ = make_model(payload["implementation"], payload["config"], device)
    candidate, implementation_sha = make_model(
        "student_stage104_gated_fast", payload["config"], device)
    reference.load_state_dict(payload["model"], strict=True)
    candidate.load_state_dict(payload["model"], strict=True)
    reference.eval(); candidate.eval()
    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        expected = reference.predict_log_probs(smoke_ids)
        actual = candidate.predict_log_probs(smoke_ids)
    max_probability_error = float((actual.exp() - expected.exp()).abs().max())
    max_logp_error = float((actual - expected).abs().max())
    max_normalization_error = float(actual.logsumexp(-1).abs().max())
    # Fusing sparse additions changes FP32 summation order. The largest
    # log-space discrepancy occurs in very low-probability tail entries;
    # compare probability mass tightly and confirm corpus NLL below.
    if (not torch.isfinite(actual).all()
            or max_probability_error > 3e-6 or max_logp_error > 2e-4
            or max_normalization_error > 1e-5):
        raise ValueError("Stage104 smoke equivalence or normalization failed: "
                         f"{max_probability_error}, {max_logp_error}, "
                         f"{max_normalization_error}")
    args.run_dir.mkdir(parents=True)
    checkpoint = args.run_dir / "stage104-gated-fast.pt"
    exported = dict(payload, implementation="student_stage104_gated_fast",
                    model=candidate.state_dict())
    exported["inference_optimization"] = dict(
        source_checkpoint_sha256=STAGE103_SHA,
        method="fused_prefix_copy_and_dynamic_collapsed_count_addition",
        no_new_gradient_targets=True,
        smoke_max_probability_error=max_probability_error,
        smoke_max_logp_error=max_logp_error,
        smoke_max_normalization_error=max_normalization_error)
    atomic_torch_save(exported, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resources_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if abs(validation["bpb"] - EXPECTED_BPB) > 2e-6:
        raise ValueError("Stage104 full validation disagrees with Stage102")
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
        protocol=PROTOCOL, status="stage104_fused_resource_audited",
        checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        source_checkpoint_sha256=STAGE103_SHA,
        validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        inference_optimization=exported["inference_optimization"],
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
