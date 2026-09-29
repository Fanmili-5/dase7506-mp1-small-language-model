"""Compare the frozen order-5 and extended order-6 counts on Stage67."""
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
from student_mixture_aware import (build_target_edge_keys,
                                   count_target_probability)
from train_experiment import atomic_json_dump

NEURAL_SHA = "3be9468b122da4c486726e8bcc59692005d0fc90dcbdd6a59c41b24d42d20b0f"
BASE_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
WEIGHTS = tuple(i / 80 for i in range(17))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if sha(args.neural) != NEURAL_SHA or sha(args.base) != BASE_SHA:
        raise ValueError("Unexpected frozen neural or base count checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    base_payload = torch.load(args.base, map_location="cpu", weights_only=True)
    extended_payload = torch.load(args.extended, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or base_payload.get("implementation") != "student_ngram"
            or extended_payload.get("implementation") != "student_ngram"
            or extended_payload["config"].get("max_order") != 6):
        raise ValueError("Unexpected Stage73 ancestry")
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(
        neural_payload["implementation"], neural_payload["config"], device
    )
    neural.load_state_dict(neural_payload["model"]); neural.eval()
    experts = {}
    for name, payload in (("order5", base_payload), ("order6", extended_payload)):
        model, _ = make_model("student_ngram", payload["config"], device)
        model.load_state_dict(payload["model"]); model.eval()
        experts[name] = (model, build_target_edge_keys(model))
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Changed validation inputs")
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(
        raw.decode("utf-8")).ids)
    totals = {name: torch.zeros(len(WEIGHTS), dtype=torch.float64)
              for name in experts}
    target_count = 0; started = time.perf_counter()
    with torch.inference_mode():
        for x, y in windows(ids, 32):
            valid = y != -100
            neural_target = neural.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)[valid].double()
            for name, (counts, edge_keys) in experts.items():
                count_target = count_target_probability(
                    counts, x, y, edge_keys
                )[valid].double().log()
                for index, weight in enumerate(WEIGHTS):
                    mixed = neural_target if weight == 0 else torch.logaddexp(
                        neural_target + math.log1p(-weight),
                        count_target + math.log(weight),
                    )
                    totals[name][index] -= mixed.sum()
            target_count += int(valid.sum())
    rows = {}
    for name, values in totals.items():
        rows[name] = [dict(
            weight=weight, nll_nats=float(value),
            bpb=float(value / math.log(2) / len(raw)),
        ) for weight, value in zip(WEIGHTS, values)]
    best = {name: min(value, key=lambda row: row["bpb"])
            for name, value in rows.items()}
    if target_count != 376599 or len(raw) != 1148007:
        raise ValueError("Coverage mismatch")
    args.run_dir.mkdir(parents=True)
    result = dict(
        protocol=PROTOCOL, split="validation", neural_sha256=NEURAL_SHA,
        base_counts_sha256=BASE_SHA, extended_counts_sha256=sha(args.extended),
        candidates=rows, best=best,
        order6_gain_bpb=best["order5"]["bpb"] - best["order6"]["bpb"],
        targets=target_count, utf8_bytes=len(raw),
        seconds=time.perf_counter() - started, no_test_scoring=True,
        note="Target-only exact screen; no candidate serialization or resource claim.",
    )
    atomic_json_dump(result, args.run_dir / "scan.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
