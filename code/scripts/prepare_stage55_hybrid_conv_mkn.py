"""Scan, freeze, serialize and resource-qualify the Stage54/MKN mixture."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows
from train_experiment import atomic_json_dump, atomic_torch_save

COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
NEURAL_SHA = "689101810f2165091647366ebaaa11e3c693edc9a80155a009237cc131a25d51"
WEIGHTS = tuple(i / 80 for i in range(17))  # 0.0000 through 0.2000
BASE_INFERENCE_FILES = (
    "student_hybrid_conv_structured.py",
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
    parser.add_argument("--expected-neural-sha", default=NEURAL_SHA)
    parser.add_argument("--source-stage", default="Stage55")
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    if (sha(args.neural) != args.expected_neural_sha or sha(args.counts) != COUNTS_SHA
            or sha(args.baseline) != BASELINE_SHA):
        raise ValueError("Unexpected frozen neural, count, or baseline checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    neural_implementation = neural_payload.get("implementation")
    supported_neural = {
        "student_hybrid_conv_structured": (
            "student_ngram_hybrid_conv_collapsed",
            ("student_ngram_hybrid_conv_collapsed.py",),
        ),
        "student_hybrid_conv_output_bias": (
            "student_ngram_hybrid_conv_bias_collapsed",
            ("student_ngram_hybrid_conv_bias_collapsed.py",
             "student_hybrid_conv_output_bias.py"),
        ),
    }
    training_provenance = (
        "training_rdrop" in neural_payload
        or "training_byte_rdrop" in neural_payload
        or neural_implementation == "student_hybrid_conv_output_bias"
    )
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_implementation not in supported_neural
            or not training_provenance
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected an R-Drop hybrid-conv neural and Stage25 MKN experts")

    device, _ = setup("cpu", "fp32", 4)
    candidate_implementation, implementation_files = supported_neural[
        neural_implementation
    ]
    inference_files = implementation_files + BASE_INFERENCE_FILES
    neural, _ = make_model(neural_implementation, neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"]); neural.eval()
    counts.load_state_dict(count_payload["model"]); counts.eval()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed validation inputs changed")
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(
        raw.decode("utf8")).ids)
    totals = torch.zeros(len(WEIGHTS), dtype=torch.float64)
    targets = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(ids, 32)):
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            neural_target = neural.predict_log_probs(x).gather(
                -1, target).squeeze(-1)[valid].double()
            count_target = counts.predict_log_probs(x).gather(
                -1, target).squeeze(-1)[valid].double()
            for index, weight in enumerate(WEIGHTS):
                if weight == 0:
                    totals[index] -= neural_target.sum()
                else:
                    totals[index] -= torch.logaddexp(
                        neural_target + math.log1p(-weight),
                        count_target + math.log(weight)).sum()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = [dict(weight=weight, nll_nats=float(total),
                 bpb=float(total / math.log(2) / len(raw)))
            for weight, total in zip(WEIGHTS, totals)]
    best = min(rows, key=lambda row: row["bpb"])
    if targets != 376599 or len(raw) != 1148007 or best["weight"] == 0:
        raise ValueError("Coverage mismatch or MKN did not improve the neural expert")
    args.run_dir.mkdir(parents=True)
    scan = dict(
        protocol=PROTOCOL, split="validation", neural_sha256=args.expected_neural_sha,
        counts_sha256=COUNTS_SHA, weights=rows, best=best, targets=targets,
        utf8_bytes=len(raw), seconds=time.perf_counter() - started,
        method="fixed_scalar_grid_before_export", no_test_scoring=True,
        source_sha256=sha(Path(__file__)),
    )
    atomic_json_dump(scan, args.run_dir / "scan.json")

    config = dict(count_payload["config"], kind="hybrid",
                  neural_config=neural_payload["config"],
                  mixture_weight=best["weight"])
    candidate, _ = make_model(candidate_implementation, config, device)
    candidate.neural.load_state_dict(neural_payload["model"])
    candidate.ngram.load_state_dict(count_payload["model"])
    candidate.eval()
    smoke_ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        actual = candidate.predict_log_probs(smoke_ids)
        expected = torch.logaddexp(
            neural.predict_log_probs(smoke_ids) + math.log1p(-best["weight"]),
            counts.predict_log_probs(smoke_ids) + math.log(best["weight"]))
    max_error = float((actual - expected).abs().max())
    if not torch.isfinite(actual).all() or max_error > 2e-5:
        raise ValueError(f"Collapsed mixture equivalence failed: {max_error}")
    exported = dict(neural_payload, implementation=candidate_implementation,
                    config=config, model=candidate.state_dict())
    exported["fixed_mixture"] = dict(
        source=f"{args.source_stage} fixed validation grid",
        neural_sha256=args.expected_neural_sha,
        counts_sha256=COUNTS_SHA, mixture_weight=best["weight"],
        selected_validation_bpb=best["bpb"], no_new_gradient_targets=True,
        smoke_max_abs_logp_error=max_error,
    )
    checkpoint = args.run_dir / "collapsed-hybrid.pt"
    atomic_torch_save(exported, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resources_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if abs(validation["bpb"] - best["bpb"]) > 2e-6:
        raise ValueError("Grid scan and collapsed CPU score disagree")
    subprocess.run([
        sys.executable, "scripts/benchmark_cpu.py",
        "--baseline", str(args.baseline.resolve()),
        "--candidate", str(checkpoint.resolve()), "--repeats", "3",
        "--threads", "4", "--output", str(resources_path.resolve()),
    ], cwd=ROOT, check=True)
    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    assets = checkpoint.stat().st_size + sum((ROOT / name).stat().st_size
                                             for name in inference_files)
    result = dict(
        protocol=PROTOCOL, status="fixed_mixture_resource_audited",
        checkpoint_sha256=sha(checkpoint), neural_sha256=args.expected_neural_sha,
        count_checkpoint_sha256=COUNTS_SHA, baseline_checkpoint_sha256=BASELINE_SHA,
        validation_bpb=validation["bpb"], cpu_ratio=resources[
            "candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        fixed_mixture=exported["fixed_mixture"], split="validation",
        precision="fp32", no_test_scoring=True,
        source_hashes={name: sha(ROOT / name) for name in inference_files},
    )
    result["qualified"] = all((result["within_five_x_time_limit"],
                                result["within_four_gib_peak_rss_limit"],
                                result["within_64mib_asset_limit"]))
    atomic_json_dump(result, args.run_dir / "final.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
