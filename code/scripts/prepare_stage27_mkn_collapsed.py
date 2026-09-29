"""Export the fixed Stage25 min2/.125 modified-KN hybrid through collapsed inference."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, make_model, setup, sha, windows
from train_experiment import atomic_json_dump, atomic_torch_save

REFERENCE_SHA = "3669552af47ef7cae72c14be15359d798ae1afbe7fbd3608f21d43414a555ad5"
EXPECTED_WEIGHT = .125
INFERENCE_FILES = (
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "student_structured.py", "student.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)
SOURCE_FILES = INFERENCE_FILES + (
    "scripts/build_kneser_ney.py", "scripts/screen_stage25_kneser_ney.py",
    "scripts/prepare_stage27_mkn_collapsed.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    if sha(args.source) != REFERENCE_SHA:
        raise ValueError("Expected the fixed Stage25 min2/.125 checkpoint")
    device, _ = setup("cpu", "fp32", 4)
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    provenance = payload.get("ancestry", {}).get("count_provenance", {})
    if (payload["protocol"] != PROTOCOL or payload["implementation"] != "student_ngram"
            or payload["config"]["mixture_weight"] != EXPECTED_WEIGHT
            or payload["config"].get("min_count") != 2
            or provenance.get("estimator") != "pruned interpolated modified Kneser-Ney"):
        raise ValueError("Unexpected Stage25 source predictor")
    old, _ = make_model("student_ngram", payload["config"], device)
    new, _ = make_model("student_ngram_collapsed", payload["config"], device)
    old.load_state_dict(payload["model"]); new.load_state_dict(payload["model"])
    old.eval(); new.eval()
    for key, value in new.state_dict().items():
        if not torch.equal(value, payload["model"][key]):
            raise ValueError("Export changed a tensor: " + key)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed data changed")
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw.decode("utf8")).ids)
    args.run_dir.mkdir(parents=True)
    started = time.perf_counter()
    totals = [0., 0.]
    targets = 0
    max_difference = 0.
    max_normalization = 0.
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(ids, 32)):
            reference, optimized = old(x), new(x)
            if reference.shape != optimized.shape or optimized.shape != (*x.shape, 2048):
                raise ValueError("Shape mismatch")
            difference = float((reference - optimized).abs().max())
            normalization = float(optimized.logsumexp(-1).abs().max())
            max_difference = max(max_difference, difference)
            max_normalization = max(max_normalization, normalization)
            if (not torch.isfinite(reference).all() or not torch.isfinite(optimized).all()
                    or difference > 2e-5 or normalization > 2e-6):
                raise ValueError(f"Full-output equivalence failed: {difference}, normalization {normalization}")
            valid = y != -100
            targets += int(valid.sum())
            for index, logp in enumerate((reference, optimized)):
                totals[index] -= logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid].double().sum().item()
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets,
                                      max_abs_logp_error=max_difference)), flush=True)
    bpb = [nll / math.log(2) / len(raw) for nll in totals]
    if targets != 376599 or len(raw) != 1148007 or abs(bpb[0] - bpb[1]) > 1e-6:
        raise ValueError("Coverage or BPB parity failed")
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during verification")
    exported = dict(payload)
    exported["implementation"] = "student_ngram_collapsed"
    exported["inference_optimization"] = dict(
        source_checkpoint_sha256=REFERENCE_SHA, source_hashes=sources,
        unchanged_tensors=True, new_training_targets=0,
        note="Same .125 mixture, modified-KN tables and Stage22 Transformer; only sparse recurrence arithmetic is collapsed.",
    )
    checkpoint = args.run_dir / "collapsed-hybrid.pt"
    atomic_torch_save(exported, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(restored["model"]) != set(payload["model"]):
        raise ValueError("Serialized state keys changed")
    if not all(torch.equal(value, restored["model"][key]) for key, value in payload["model"].items()):
        raise ValueError("Serialized tensors changed")
    result = dict(
        protocol=PROTOCOL, split="validation", device="cpu", precision="fp32", threads=4,
        reference_sha256=REFERENCE_SHA, checkpoint_sha256=sha(checkpoint), source_hashes=sources,
        targets=targets, utf8_bytes=len(raw), old_bpb=bpb[0], optimized_bpb=bpb[1],
        max_abs_logp_error=max_difference,
        max_abs_log_normalization_error=max_normalization,
        positive_count_floor=new.count_probability_lower_bound,
        unchanged_serialized_tensors=True, new_gradient_targets=0,
        asset_bytes=checkpoint.stat().st_size + sum((ROOT / name).stat().st_size for name in INFERENCE_FILES),
        equivalence_seconds=time.perf_counter() - started,
        status="full_validation_equivalence_passed_resource_measurement_pending",
        note="No test scoring or new selection; dual-model parity is not a timing gate.",
    )
    atomic_json_dump(result, args.run_dir / "equivalence.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
