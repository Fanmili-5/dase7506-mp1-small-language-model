"""Serialize and verify the fixed Stage32 calibration with collapsed MKN inference."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows
from train_experiment import atomic_json_dump, atomic_torch_save

SOURCE_SHA = "3847ff751554be8f22e71882046e91924db590b2bac4df1ac489796625fee817"
EXPECTED_BPB = 1.4464619984403881
TEMPERATURE = 1.075
PRIOR_WEIGHT = .05
GATE_SHIFT = .25
MIXTURE_WEIGHT = .1125
INFERENCE_FILES = (
    "student_calibrated_collapsed.py", "student_ngram_collapsed.py",
    "student_ngram_fast.py", "student_ngram.py", "student_structured.py", "student.py",
    "common.py", "evaluate.py", "data/tokenizer.json", "requirements.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    if sha(args.source) != SOURCE_SHA:
        raise ValueError("Unexpected Stage30 source checkpoint")
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_ngram":
        raise ValueError("Unexpected source ancestry")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    train_text = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    train_ids = torch.tensor(tokenizer.encode(train_text).ids)
    frequencies = torch.bincount(train_ids, minlength=2048).double() + .1
    log_prior = (frequencies / frequencies.sum()).log().float()
    config = dict(payload["config"], kind="hybrid_calibrated",
                  mixture_weight=MIXTURE_WEIGHT, vocabulary_temperature=TEMPERATURE,
                  unigram_prior_weight=PRIOR_WEIGHT, copy_gate_shift=GATE_SHIFT,
                  calibration_log_prior=log_prior.tolist())
    device, _ = setup("cpu", "fp32", 4)
    old, _ = make_model("student_ngram", payload["config"], device)
    new, _ = make_model("student_calibrated_collapsed", config, device)
    old.load_state_dict(payload["model"]); old.eval()
    new.neural.load_state_dict(old.neural.state_dict())
    new.ngram.load_state_dict(old.ngram.state_dict()); new.eval()
    validation_raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    validation_ids = torch.tensor(tokenizer.encode(validation_raw.decode("utf-8")).ids)
    totals = [0., 0.]
    targets = 0
    max_difference = 0.
    max_normalization = 0.
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(validation_ids, 32)):
            hidden = old.neural.features(x).float()
            vocabulary = F.softmax(old.neural.head(hidden) / TEMPERATURE
                                   + PRIOR_WEIGHT * log_prior, dim=-1)
            copy = old.neural.copy_distribution(hidden, x)
            gate = old.neural.copy_gate(hidden) + GATE_SHIFT
            reference = ((1 - MIXTURE_WEIGHT) * (torch.sigmoid(-gate) * vocabulary
                         + torch.sigmoid(gate) * copy)
                         + MIXTURE_WEIGHT * old.ngram.distribution(x))
            reference.div_(reference.sum(-1, keepdim=True))
            reference = reference.log()
            optimized = new.predict_log_probs(x)
            difference = float((reference - optimized).abs().max())
            normalization = float(optimized.logsumexp(-1).abs().max())
            max_difference = max(max_difference, difference)
            max_normalization = max(max_normalization, normalization)
            if difference > 2e-5 or normalization > 2e-6:
                raise ValueError("Full-output calibration equivalence failed")
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            for index, logp in enumerate((reference, optimized)):
                totals[index] -= logp.gather(-1, target).squeeze(-1)[valid].double().sum().item()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets,
                                      max_abs_logp_error=max_difference)), flush=True)
    bpbs = [value / math.log(2) / len(validation_raw) for value in totals]
    if (targets != 376599 or len(validation_raw) != 1148007
            or abs(bpbs[0] - EXPECTED_BPB) > 2e-5 or abs(bpbs[0] - bpbs[1]) > 1e-6):
        raise ValueError("Coverage, selected-score, or BPB parity failed")
    args.run_dir.mkdir(parents=True)
    exported = dict(payload, implementation="student_calibrated_collapsed", config=config,
                    model=new.state_dict())
    exported["calibration"] = dict(
        source_checkpoint_sha256=SOURCE_SHA, temperature=TEMPERATURE,
        unigram_prior_weight=PRIOR_WEIGHT, copy_gate_shift=GATE_SHIFT,
        mixture_weight=MIXTURE_WEIGHT, train_unigram_tokens=len(train_ids),
        no_new_gradient_targets=True,
    )
    checkpoint = args.run_dir / "calibrated-collapsed.pt"
    atomic_torch_save(exported, checkpoint)
    sources = {name: sha(ROOT / name) for name in INFERENCE_FILES}
    result = dict(
        protocol=PROTOCOL, split="validation", reference_sha256=SOURCE_SHA,
        checkpoint_sha256=sha(checkpoint), reference_bpb=bpbs[0], optimized_bpb=bpbs[1],
        targets=targets, utf8_bytes=len(validation_raw), max_abs_logp_error=max_difference,
        max_abs_log_normalization_error=max_normalization,
        conservative_asset_bytes=checkpoint.stat().st_size
                                 + sum((ROOT / name).stat().st_size for name in INFERENCE_FILES),
        source_hashes=sources, seconds=time.perf_counter() - started,
        status="calibrated_full_validation_equivalence_passed_resources_pending",
        no_test_scoring=True,
    )
    atomic_json_dump(result, args.run_dir / "equivalence.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
