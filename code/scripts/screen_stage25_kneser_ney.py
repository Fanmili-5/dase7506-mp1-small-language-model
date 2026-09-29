"""Scan a fixed neural/modified-KN mixture grid on validation."""
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
from student_ngram import build_model
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload

NEURAL_SHA = "05f20e68313a3c41c4a1116bc3d335ad688507c933fb60e9176a83d8953e5269"
WEIGHTS = (0., .025, .05, .075, .10, .125, .15, .20, .25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if sha(args.neural) != NEURAL_SHA:
        raise ValueError("Unexpected Stage22 neural checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    provenance = count_payload.get("statistics_provenance", {})
    if (neural_payload["protocol"] != PROTOCOL or neural_payload["implementation"] != "student_structured"
            or "training_deep_supervision" not in neural_payload
            or count_payload["protocol"] != PROTOCOL or count_payload["implementation"] != "student_ngram"
            or provenance.get("estimator") != "pruned interpolated modified Kneser-Ney"):
        raise ValueError("Unexpected neural/count ancestry")
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model("student_structured", neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"]); neural.eval()
    counts.load_state_dict(count_payload["model"]); counts.eval()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_validation.txt", "tokenizer.json", "wikitext_train.txt"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed data changed")
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw.decode("utf8")).ids)
    totals = {weight: 0. for weight in WEIGHTS}
    targets = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(ids, 32)):
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            neural_target = neural.predict_log_probs(x).gather(-1, target).squeeze(-1)[valid]
            count_target = counts.predict_log_probs(x).gather(-1, target).squeeze(-1)[valid]
            for weight in WEIGHTS:
                mixed = neural_target if weight == 0 else torch.logaddexp(
                    neural_target + math.log1p(-weight), count_target + math.log(weight))
                totals[weight] -= mixed.double().sum().item()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = [dict(weight=weight, nll_nats=totals[weight],
                 bpb=totals[weight] / math.log(2) / len(raw)) for weight in WEIGHTS]
    best = min(rows, key=lambda row: row["bpb"])
    if targets != 376599 or len(raw) != 1148007:
        raise ValueError("Coverage mismatch")
    args.run_dir.mkdir(parents=True)
    count_sha = sha(args.counts)
    result = dict(protocol=PROTOCOL, split="validation", neural_sha256=NEURAL_SHA,
                  counts_sha256=count_sha, count_config=count_payload["config"],
                  count_provenance=provenance, candidates=rows, best=best,
                  targets=targets, utf8_bytes=len(raw), seconds=time.perf_counter() - started,
                  new_gradient_targets=0,
                  note="Fixed validation grid; modified-KN statistics are train-only. No test scoring.")
    if best["weight"] > 0:
        config = dict(count_payload["config"], kind="hybrid",
                      neural_config=neural_payload["config"], mixture_weight=best["weight"])
        hybrid = build_model(config).eval()
        hybrid.neural.load_state_dict(neural_payload["model"])
        hybrid.ngram.load_state_dict(count_payload["model"])
        payload = checkpoint_payload(hybrid, "student_ngram", config, 17,
                                     neural_payload.get("train_tokens", 58982400))
        payload["ancestry"] = dict(
            neural_sha256=NEURAL_SHA, counts_sha256=count_sha,
            neural_training_targets=neural_payload.get("train_tokens"),
            training_deep_supervision=neural_payload["training_deep_supervision"],
            count_provenance=provenance,
        )
        checkpoint = args.run_dir / "best-mixture.pt"
        atomic_torch_save(payload, checkpoint)
        output = args.run_dir / "best-validation-cpu-fp32.json"
        subprocess.run([sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
                        "--device", "cpu", "--precision", "fp32", "--threads", "4",
                        "--split", "validation", "--output", str(output.resolve())],
                       cwd=ROOT, check=True)
        official = json.loads(output.read_text(encoding="utf-8"))
        if abs(official["bpb"] - best["bpb"]) > 1e-6:
            raise ValueError("Scan/official mismatch")
        result.update(checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                      official_bpb=official["bpb"])
    atomic_json_dump(result, args.run_dir / "scan.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
