"""Serialize and resource-qualify the final Stage79 scalar calibration."""
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
from torch.nn import functional as F
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha
from train_experiment import atomic_json_dump, atomic_torch_save


NEURAL_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
INFERENCE_FILES = (
    "student_ngram_hybrid_conv_calibrated_collapsed.py",
    "student_ngram_hybrid_conv_bias_collapsed.py",
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "student_structured.py", "student.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if (sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA
            or sha(args.baseline) != BASELINE_SHA):
        raise ValueError("Unexpected frozen neural, count, or baseline checkpoint")

    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    if (screen.get("protocol") != PROTOCOL
            or screen.get("split") != "validation"
            or screen.get("neural_sha256") != NEURAL_SHA
            or screen.get("counts_sha256") != COUNTS_SHA
            or not screen.get("scalar_calibration_closed_after_this_scan")
            or not screen.get("no_test_scoring")):
        raise ValueError("Expected the closed Stage79 validation screen")
    best = screen["best"]
    temperature = float(best["temperature"])
    prior_weight = float(best["unigram_prior_weight"])
    gate_shift = float(best["copy_gate_shift"])
    mixture = float(best["mixture_weight"])
    if not (temperature > 0 and 0 < mixture < 1
            and all(math.isfinite(value) for value in
                    (temperature, prior_weight, gate_shift, mixture))):
        raise ValueError("Invalid selected scalar calibration")

    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected the Stage71 and Stage25 expert payloads")

    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_train.txt", "wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed data changed")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    train_text = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    train_ids = torch.tensor(tokenizer.encode(train_text).ids)
    frequencies = torch.bincount(train_ids, minlength=2048).double() + .1
    log_prior = (frequencies / frequencies.sum()).log().float()

    device, _ = setup("cpu", "fp32", 4)
    original, _ = make_model(neural_payload["implementation"],
                             neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    original.load_state_dict(neural_payload["model"]); original.eval()
    counts.load_state_dict(count_payload["model"]); counts.eval()
    config = dict(
        count_payload["config"], kind="hybrid_calibrated_conv",
        neural_config=neural_payload["config"], mixture_weight=mixture,
        vocabulary_temperature=temperature,
    )
    candidate, _ = make_model(
        "student_ngram_hybrid_conv_calibrated_collapsed", config, device)
    candidate.neural.load_state_dict(neural_payload["model"])
    candidate.ngram.load_state_dict(count_payload["model"])
    old_bias = candidate.neural.output_bias.detach().clone()
    with torch.no_grad():
        candidate.neural.output_bias.copy_(
            old_bias / temperature + prior_weight * log_prior)
        candidate.neural.copy_gate.bias.add_(gate_shift)
    candidate.eval()

    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        hidden = original.features(smoke_ids).float()
        vocabulary = F.softmax(
            (original.head(hidden) + original.output_bias) / temperature
            + prior_weight * log_prior,
            dim=-1,
        )
        copy = original.copy_distribution(hidden, smoke_ids)
        gate = original.copy_gate(hidden) + gate_shift
        expected = vocabulary * (torch.sigmoid(-gate) * (1 - mixture))
        expected.addcmul_(copy, torch.sigmoid(gate) * (1 - mixture))
        candidate.ngram.add_into(expected, smoke_ids, mixture)
        expected.div_(expected.sum(-1, keepdim=True))
        actual = candidate.predict_log_probs(smoke_ids)
    max_error = float((actual.exp() - expected).abs().max())
    if (not torch.isfinite(actual).all() or max_error > 3e-6
            or float(actual.logsumexp(-1).abs().max()) > 3e-6):
        raise ValueError(f"Materialized calibration equivalence failed: {max_error}")

    args.run_dir.mkdir(parents=True)
    exported = dict(
        neural_payload, implementation="student_ngram_hybrid_conv_calibrated_collapsed",
        config=config, model=candidate.state_dict(),
    )
    exported["fixed_calibration"] = dict(
        source="Stage79 closed validation grid", screen_sha256=sha(args.screen),
        neural_sha256=NEURAL_SHA, counts_sha256=COUNTS_SHA,
        vocabulary_temperature=temperature, unigram_prior_weight=prior_weight,
        copy_gate_shift=gate_shift, mixture_weight=mixture,
        selected_validation_bpb=float(best["bpb"]),
        train_unigram_tokens=len(train_ids), no_new_gradient_targets=True,
        smoke_max_abs_probability_error=max_error,
    )
    checkpoint = args.run_dir / "calibrated-collapsed-hybrid.pt"
    atomic_torch_save(exported, checkpoint)

    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resources_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if abs(validation["bpb"] - float(best["bpb"])) > 2e-6:
        raise ValueError("Stage79 screen and materialized CPU score disagree")
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
        protocol=PROTOCOL, status="fixed_calibration_resource_audited",
        checkpoint_sha256=sha(checkpoint), neural_sha256=NEURAL_SHA,
        count_checkpoint_sha256=COUNTS_SHA,
        baseline_checkpoint_sha256=BASELINE_SHA,
        validation_bpb=validation["bpb"],
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        fixed_calibration=exported["fixed_calibration"], split="validation",
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
