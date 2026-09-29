"""Export Stage37 through temperature-folded inference with full validation parity."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows
from train_experiment import atomic_json_dump, atomic_torch_save

SOURCE_SHA = "ed29352b1fca4c325beec7fcb30ab2b4a019a3f264b256945d119534c2380e43"
INFERENCE_FILES = (
    "student_calibrated_collapsed_fast.py", "student_calibrated_collapsed.py",
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "student_structured.py", "student.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    if sha(args.source) != SOURCE_SHA:
        raise ValueError("Unexpected exact Stage37 checkpoint")
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["implementation"] != "student_calibrated_collapsed":
        raise ValueError("Unexpected Stage37 ancestry")
    device, _ = setup("cpu", "fp32", 4)
    exact, _ = make_model(payload["implementation"], payload["config"], device)
    exact.load_state_dict(payload["model"]); exact.eval()
    config = dict(payload["config"], kind="hybrid_calibrated_fast")
    fast, _ = make_model("student_calibrated_collapsed_fast", config, device)
    fast.load_state_dict(payload["model"]); fast.eval()
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(tokenizer.encode(raw.decode("utf-8")).ids)
    totals = [0., 0.]
    targets = 0
    max_difference = 0.
    max_normalization = 0.
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(ids, 32)):
            reference, optimized = exact(x), fast(x)
            difference = float((reference - optimized).abs().max())
            normalization = float(optimized.logsumexp(-1).abs().max())
            max_difference = max(max_difference, difference)
            max_normalization = max(max_normalization, normalization)
            if difference > 5e-5 or normalization > 2e-6:
                raise ValueError(f"Fast calibration parity failed: {difference}, {normalization}")
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            for index, logp in enumerate((reference, optimized)):
                totals[index] -= logp.gather(-1, target).squeeze(-1)[valid].double().sum().item()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets,
                                      max_abs_logp_error=max_difference)), flush=True)
    bpbs = [value / math.log(2) / len(raw) for value in totals]
    if targets != 376599 or len(raw) != 1148007 or abs(bpbs[0] - bpbs[1]) > 1e-6:
        raise ValueError("Coverage or complete-score parity failed")
    args.run_dir.mkdir(parents=True)
    exported = dict(payload, implementation="student_calibrated_collapsed_fast",
                    config=config, model=fast.state_dict())
    exported["inference_optimization"] = dict(
        source_checkpoint_sha256=SOURCE_SHA, unchanged_state_tensors=True,
        operation="Move scalar temperature division before bias-free vocabulary projection",
        no_new_training_targets=True,
    )
    checkpoint = args.run_dir / "calibrated-collapsed.pt"
    atomic_torch_save(exported, checkpoint)
    sources = {name: sha(ROOT / name) for name in INFERENCE_FILES}
    result = dict(
        protocol=PROTOCOL, split="validation", reference_sha256=SOURCE_SHA,
        checkpoint_sha256=sha(checkpoint), reference_bpb=bpbs[0], optimized_bpb=bpbs[1],
        targets=targets, utf8_bytes=len(raw), max_abs_logp_error=max_difference,
        max_abs_log_normalization_error=max_normalization,
        conservative_asset_bytes=checkpoint.stat().st_size
                                 + sum((ROOT / name).stat().st_size for name in INFERENCE_FILES),
        source_hashes=sources, seconds=time.perf_counter() - started,
        status="fast_calibrated_full_validation_equivalence_passed_resources_pending",
        no_test_scoring=True,
    )
    atomic_json_dump(result, args.run_dir / "equivalence.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
