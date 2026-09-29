"""Validation-only ceiling analysis for neural/count mixing; never exports a model.

Target-conditioned oracle rows are explicitly illegal predictors. They answer
whether a dynamic gate could in principle bridge the requested quality gap.
"""
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
from train_experiment import atomic_json_dump

NEURAL_SHA = "2be8f4f4ac195593835aaa74f042b0b5d0f5473a46eb6d09fba99b3c11d263f7"
COUNTS_SHA = "b1898559c73ccf62e6e945c1a9269e6fe2b0ee4499b5a1d88e7e30020c194230"
WEIGHTS = tuple(i / 40 for i in range(41))  # fixed 0,.025,...,1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neural", type=Path, required=True)
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output must be new")
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Use the fixed H and train-count checkpoints")
    device, _ = setup("cpu", "fp32", 4)
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    neural, _ = make_model(neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model(count_payload["implementation"], count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"]); neural.eval()
    counts.load_state_dict(count_payload["model"]); counts.eval()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed validation inputs changed")
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw.decode("utf8")).ids)
    totals = torch.zeros(len(WEIGHTS), dtype=torch.float64)
    oracle_nll = 0.0
    neural_wins = count_wins = targets = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(ids, 32)):
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            a = neural.predict_log_probs(x).gather(-1, target).squeeze(-1)[valid].double()
            b = counts.predict_log_probs(x).gather(-1, target).squeeze(-1)[valid].double()
            stacked = []
            for w in WEIGHTS:
                if w == 0:
                    mixed = a
                elif w == 1:
                    mixed = b
                else:
                    mixed = torch.logaddexp(a + math.log1p(-w), b + math.log(w))
                stacked.append(mixed)
            grid = torch.stack(stacked)
            totals -= grid.sum(1)
            # This sees the correct target and is an unattainable diagnostic ceiling.
            oracle_nll -= grid.max(0).values.sum().item()
            neural_wins += int((a >= b).sum())
            count_wins += int((b > a).sum())
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    bpbs = [float(v / math.log(2) / len(raw)) for v in totals]
    rows = [dict(weight=w, bpb=bpb) for w, bpb in zip(WEIGHTS, bpbs)]
    best = min(rows, key=lambda r: r["bpb"])
    result = dict(protocol=PROTOCOL, split="validation", purpose="diagnostic_ceiling_only",
        neural_sha256=NEURAL_SHA, counts_sha256=COUNTS_SHA, targets=targets,
        utf8_bytes=len(raw), fixed_grid=rows, best_fixed=best,
        target_oracle_grid_bpb=oracle_nll / math.log(2) / len(raw),
        target_oracle_is_illegal=True, neural_target_wins=neural_wins,
        count_target_wins=count_wins, requested_threshold=1.4,
        seconds=time.perf_counter()-started,
        interpretation="Oracle uses answers and can never be deployed. It is only an upper bound on gates restricted to these two unchanged experts.",
        source_hashes={name: sha(ROOT / name) for name in
            ("scripts/analyze_stage22_gate_ceiling.py", "student_ngram.py", "student_structured.py",
             "student.py", "common.py", "data/tokenizer.json")})
    if targets != 376599 or len(raw) != 1148007:
        raise ValueError("Validation coverage mismatch")
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
