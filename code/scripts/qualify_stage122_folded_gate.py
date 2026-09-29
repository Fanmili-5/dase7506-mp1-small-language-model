"""Export the exact Stage115 gate with folded calibration and audit CPU resources."""
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
from scripts.probe_stage116_rowmax import install_derived_buffers
from train_experiment import atomic_json_dump, atomic_torch_save


STAGE115_SHA = "902e4b21c9ddf3afda2258ae032fe852517716cccfd5341567b2fabc21910e76"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
EXPECTED_BPB = 1.40022494630321
IMPLEMENTATION = "student_stage122_folded_gate"
INFERENCE_FILES = (
    "student_stage122_folded_gate.py", "student_stage116_rowmax_gate.py",
    "student_stage115_order5_gate.py", "student_stage105_gated_singlepass.py",
    "student_stage104_gated_fast.py", "student_stage103_gated.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram_collapsed.py",
    "student_ngram_fast.py", "student_ngram.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage115", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage122 output directory")
    if sha(args.stage115) != STAGE115_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Unexpected Stage115 or baseline checkpoint")
    payload = torch.load(args.stage115, map_location="cpu", weights_only=True)
    if payload.get("protocol") != PROTOCOL or payload.get("implementation") != "student_stage115_order5_gate":
        raise ValueError("Unexpected Stage115 checkpoint format")
    device, _ = setup("cpu", "fp32", 4)
    reference, _ = make_model(payload["implementation"], payload["config"], device)
    candidate, _ = make_model(IMPLEMENTATION, payload["config"], device)
    reference.load_state_dict(payload["model"], strict=True)
    mismatch = candidate.load_state_dict(payload["model"], strict=False)
    missing = {f"ngram.tables.{i}.row_max" for i in range(4)}
    missing |= {"gate_alpha", "gate_intercept"}
    if set(mismatch.missing_keys) != missing or mismatch.unexpected_keys:
        raise ValueError("Derived-buffer state mismatch")
    install_derived_buffers(candidate)
    temperature = float(candidate.temperature)
    with torch.no_grad():
        candidate.neural.norm.weight.div_(temperature)
        if getattr(candidate.neural.norm, "bias", None) is not None:
            candidate.neural.norm.bias.div_(temperature)
        candidate.neural.output_bias.div_(temperature)
        candidate.neural.output_bias.add_(candidate.prior_weight * candidate.log_prior)
        candidate.neural.copy_query.weight.mul_(temperature)
        candidate.neural.copy_key.weight.mul_(temperature)
        candidate.neural.copy_gate.weight.mul_(temperature)
        candidate.neural.copy_gate.bias.add_(candidate.copy_shift)
    reference.eval(); candidate.eval()
    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        expected = reference.predict_log_probs(smoke_ids)
        actual = candidate.predict_log_probs(smoke_ids)
    smoke = dict(max_probability_error=float((actual.exp() - expected.exp()).abs().max()),
                 max_logp_error=float((actual - expected).abs().max()),
                 max_normalization_error=float(actual.logsumexp(-1).abs().max()))
    if (not torch.isfinite(actual).all() or smoke["max_probability_error"] > 3e-6
            or smoke["max_logp_error"] > 3e-4
            or smoke["max_normalization_error"] > 1e-5):
        raise ValueError(f"Stage122 folded smoke mismatch: {smoke}")
    args.run_dir.mkdir(parents=True)
    checkpoint = args.run_dir / "stage122-folded-gate.pt"
    exported = dict(payload, implementation=IMPLEMENTATION,
                    model=candidate.state_dict())
    exported["inference_optimization"] = dict(
        source_checkpoint_sha256=STAGE115_SHA,
        method="fold_stage94_affines_cache_count_rowmax_fold_gate_standardization",
        no_new_training_targets=True, no_test_scoring=True, smoke=smoke)
    atomic_torch_save(exported, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resource_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if abs(validation["bpb"] - EXPECTED_BPB) > 2e-6:
        raise ValueError("Stage122 full validation disagrees with Stage115")
    subprocess.run([
        sys.executable, "scripts/benchmark_cpu.py", "--baseline",
        str(args.baseline.resolve()), "--candidate", str(checkpoint.resolve()),
        "--repeats", "3", "--threads", "4", "--output",
        str(resource_path.resolve()),
    ], cwd=ROOT, check=True)
    resources = json.loads(resource_path.read_text(encoding="utf-8"))
    assets = checkpoint.stat().st_size + sum(
        (ROOT / file).stat().st_size for file in INFERENCE_FILES)
    final = dict(protocol=PROTOCOL, status="stage122_folded_gate_resource_audited",
                 checkpoint_sha256=sha(checkpoint),
                 checkpoint_bytes=checkpoint.stat().st_size,
                 validation_bpb=validation["bpb"],
                 cpu_ratio=resources["candidate_to_baseline_time_ratio"],
                 peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
                 conservative_asset_bytes=assets,
                 within_five_x_time_limit=resources["within_five_x_time_limit"],
                 within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
                 within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
                 inference_optimization=exported["inference_optimization"],
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
