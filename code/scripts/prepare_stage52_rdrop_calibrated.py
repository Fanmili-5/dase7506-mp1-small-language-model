"""Apply the already-frozen Stage32 calibration to Stage47 and qualify it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, sha
from train_experiment import atomic_json_dump, atomic_torch_save

COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
TEMPERATURE = 1.075
PRIOR_WEIGHT = .05
GATE_SHIFT = .25
MIXTURE_WEIGHT = .1125
INFERENCE_FILES = (
    "student_calibrated_collapsed_fast.py", "student_calibrated_collapsed.py",
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
        parser.error("Use a new run directory")
    neural_sha = sha(args.neural)
    if sha(args.counts) != COUNTS_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Unexpected fixed count or baseline checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_structured"
            or "training_rdrop" not in neural_payload
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected exported Stage47 neural and fixed Stage25 counts")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    train_text = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    train_ids = torch.tensor(tokenizer.encode(train_text).ids)
    frequencies = torch.bincount(train_ids, minlength=2048).double() + .1
    log_prior = (frequencies / frequencies.sum()).log().float()
    config = dict(
        count_payload["config"], kind="hybrid_calibrated_fast",
        neural_config=neural_payload["config"], mixture_weight=MIXTURE_WEIGHT,
        vocabulary_temperature=TEMPERATURE, unigram_prior_weight=PRIOR_WEIGHT,
        copy_gate_shift=GATE_SHIFT, calibration_log_prior=log_prior.tolist(),
    )
    candidate, _ = make_model(
        "student_calibrated_collapsed_fast", config, torch.device("cpu"))
    candidate.neural.load_state_dict(neural_payload["model"])
    candidate.ngram.load_state_dict(count_payload["model"])
    candidate.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        logp = candidate.predict_log_probs(ids)
    if not torch.isfinite(logp).all() or float(logp.logsumexp(-1).abs().max()) > 2e-6:
        raise ValueError("Calibrated candidate failed finite normalization smoke test")
    args.run_dir.mkdir(parents=True)
    exported = dict(
        neural_payload, implementation="student_calibrated_collapsed_fast",
        config=config, model=candidate.state_dict(),
    )
    exported["fixed_calibration"] = dict(
        source="Stage32 settings selected before Stage47 existed",
        neural_sha256=neural_sha, counts_sha256=COUNTS_SHA,
        vocabulary_temperature=TEMPERATURE, unigram_prior_weight=PRIOR_WEIGHT,
        copy_gate_shift=GATE_SHIFT, mixture_weight=MIXTURE_WEIGHT,
        train_unigram_tokens=len(train_ids), no_new_gradient_targets=True,
        no_stage47_specific_validation_selection=True,
    )
    checkpoint = args.run_dir / "calibrated-collapsed.pt"
    atomic_torch_save(exported, checkpoint)
    validation_path = args.run_dir / "validation-cpu-fp32.json"
    resources_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
        "--device", "cpu", "--precision", "fp32", "--threads", "4",
        "--split", "validation", "--output", str(validation_path.resolve()),
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "scripts/benchmark_cpu.py", "--baseline", str(args.baseline.resolve()),
        "--candidate", str(checkpoint.resolve()), "--repeats", "3", "--threads", "4",
        "--output", str(resources_path.resolve()),
    ], cwd=ROOT, check=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    assets = checkpoint.stat().st_size + sum((ROOT / name).stat().st_size for name in INFERENCE_FILES)
    result = dict(
        protocol=PROTOCOL, status="fixed_calibration_resource_audited",
        checkpoint_sha256=sha(checkpoint), neural_sha256=neural_sha,
        count_checkpoint_sha256=COUNTS_SHA, baseline_checkpoint_sha256=BASELINE_SHA,
        validation_bpb=validation["bpb"], cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        conservative_asset_bytes=assets,
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        fixed_calibration=exported["fixed_calibration"], split="validation",
        precision="fp32", no_test_scoring=True,
        source_hashes={name: sha(ROOT / name) for name in INFERENCE_FILES},
    )
    result["qualified"] = all((result["within_five_x_time_limit"],
                                result["within_four_gib_peak_rss_limit"],
                                result["within_64mib_asset_limit"]))
    atomic_json_dump(result, args.run_dir / "final.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
