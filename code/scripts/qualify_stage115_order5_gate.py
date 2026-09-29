"""Export the Stage114 five-order gate and preflight official CPU resources."""
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
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA
from scripts.fit_stage100_train_gate import BASE_SHA, FEATURE_NAMES
from scripts.screen_stage110_low_cost_gate import STAGE100_SHA
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload


EXPECTED_BPB = 1.4002249460704805
SELECTED_NAMES = (
    "neural_max_logp", "neural_margin", "highest_backoff", "highest_max_mass")
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
INFERENCE_FILES = (
    "student_stage115_order5_gate.py", "student_stage105_gated_singlepass.py",
    "student_stage104_gated_fast.py", "student_stage103_gated.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_ngram_collapsed.py",
    "student_ngram_fast.py", "student_ngram.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage115 output directory")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != BASE_SHA
            or sha(args.train_gate) != STAGE100_SHA
            or sha(args.baseline) != BASELINE_SHA):
        raise ValueError("Unexpected fixed expert, gate, or baseline")
    selection_sha = sha(args.selection)
    selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
    selected = selection["best"]
    if (selection.get("no_validation_gradient_fitting") is not True
            or selected.get("variant") != "full_reference"
            or selected.get("active_features") != list(SELECTED_NAMES)
            or selected.get("slope_scale") != .4
            or abs(selected.get("bpb") - EXPECTED_BPB) > 1e-10):
        raise ValueError("Stage114 setting has changed")
    gate = json.loads(args.train_gate.read_text(encoding="utf-8-sig"))
    if (gate.get("protocol") != PROTOCOL
            or gate.get("no_validation_or_test_fitting") is not True
            or gate.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Expected training-derived gate coefficients")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"
            or count_payload["config"].get("max_order") != 5):
        raise ValueError("Unexpected expert checkpoint format")
    config = dict(count_payload["config"], kind="gated_hybrid",
                  neural_config=neural_payload["config"],
                  feature_names=list(SELECTED_NAMES),
                  vocabulary_temperature=1.125,
                  train_unigram_prior_weight=.0625,
                  copy_gate_shift=.25, anchor_weight=.0625, slope_scale=.4)
    device, _ = setup("cpu", "fp32", 4)
    model, _ = make_model("student_stage115_order5_gate", config, device)
    model.neural.load_state_dict(neural_payload["model"], strict=True)
    model.ngram.load_state_dict(count_payload["model"], strict=True)
    log_prior, train_tokens = train_log_prior()
    indices = [FEATURE_NAMES.index(name) for name in SELECTED_NAMES]
    with torch.no_grad():
        model.log_prior.copy_(log_prior)
        model.gate_mean.copy_(torch.tensor(
            [gate["feature_mean"][i] for i in indices], dtype=torch.float64))
        model.gate_std.copy_(torch.tensor(
            [gate["feature_std"][i] for i in indices], dtype=torch.float64))
        model.gate_coeff.copy_(torch.tensor(
            [gate["coefficients"][name] for name in SELECTED_NAMES],
            dtype=torch.float64))
    model.eval()
    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        smoke = model.predict_log_probs(smoke_ids)
    normalization_error = float(smoke.logsumexp(-1).abs().max())
    if not torch.isfinite(smoke).all() or normalization_error > 1e-5:
        raise ValueError("Five-order predictor failed normalization smoke")
    args.run_dir.mkdir(parents=True)
    checkpoint = args.run_dir / "stage115-order5-gate.pt"
    payload = checkpoint_payload(model, "student_stage115_order5_gate", config,
                                 int(neural_payload.get("seed", 92017)),
                                 int(neural_payload.get("train_tokens", 0)))
    payload["ancestry"] = dict(
        neural_sha256=NEURAL_SHA, counts_sha256=BASE_SHA,
        train_gate_sha256=STAGE100_SHA, selection_sha256=selection_sha,
        train_unigram_tokens=train_tokens, no_test_scoring=True)
    atomic_torch_save(payload, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resource_path = args.run_dir / "resources-one-repeat.json"
    subprocess.run([sys.executable, "evaluate.py", "--checkpoint",
                    str(checkpoint.resolve()), "--device", "cpu", "--precision",
                    "fp32", "--threads", "4", "--split", "validation", "--output",
                    str(validation_path.resolve())], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if abs(validation["bpb"] - EXPECTED_BPB) > 2e-6:
        raise ValueError("Exact Stage115 export disagrees with Stage114")
    subprocess.run([sys.executable, "scripts/benchmark_cpu.py", "--baseline",
                    str(args.baseline.resolve()), "--candidate",
                    str(checkpoint.resolve()), "--repeats", "1", "--threads", "4",
                    "--output", str(resource_path.resolve())], cwd=ROOT, check=True)
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    assets = checkpoint.stat().st_size + sum((ROOT / file).stat().st_size
                                              for file in INFERENCE_FILES)
    result = dict(
        protocol=PROTOCOL, status="one_repeat_resource_preflight_not_qualification",
        checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        validation_bpb=validation["bpb"], cpu_ratio_one_repeat=
        resource["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resource["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_one_repeat=resource["within_five_x_time_limit"],
        within_four_gib=resource["within_four_gib_peak_rss_limit"],
        within_64mib=assets <= 64 * 1024 ** 2,
        smoke_max_normalization_error=normalization_error,
        inference_files=INFERENCE_FILES,
        source_hashes={file: sha(ROOT / file) for file in INFERENCE_FILES},
        split="validation", precision="fp32", no_test_scoring=True)
    atomic_json_dump(result, args.run_dir / "preflight.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
