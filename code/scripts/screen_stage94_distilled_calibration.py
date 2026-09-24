"""Bounded scalar recalibration of the fixed Stage92 distilled average."""
from __future__ import annotations

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
from train_experiment import atomic_json_dump


NEURAL_SHA = "5e4ca423aeca652f5193f6b6b93eafe2a940ba6bab08689cb56b09c5c0b4c417"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
REFERENCE_BPB = 1.4017970556559118
TEMPERATURES = (1.05, 1.075, 1.10, 1.125, 1.15)
PRIOR_WEIGHTS = (.025, .0375, .05, .0625, .075)
COPY_GATE_SHIFTS = (.0625, .125, .1875, .25, .3125)
MIXTURE_WEIGHTS = (.0375, .05, .0625, .075, .0875, .10, .1125)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new output directory")
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected frozen Stage92 or Stage25 checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected expert payload")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True); neural.eval()
    counts.load_state_dict(count_payload["model"], strict=True); counts.eval()

    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_train.txt", "wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed data changed")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    train_text = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    train_ids = torch.tensor(tokenizer.encode(train_text).ids)
    frequencies = torch.bincount(train_ids, minlength=2048).double() + .1
    log_prior = (frequencies / frequencies.sum()).log().float().to(device)
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    validation_ids = torch.tensor(tokenizer.encode(raw.decode("utf-8")).ids)
    grid = [(temperature, prior, gate, mixture)
            for temperature in TEMPERATURES for prior in PRIOR_WEIGHTS
            for gate in COPY_GATE_SHIFTS for mixture in MIXTURE_WEIGHTS]
    totals = torch.zeros(len(grid), dtype=torch.float64, device=device)
    mixture_tensor = torch.tensor(MIXTURE_WEIGHTS, device=device)
    targets = 0; started = time.perf_counter()
    with torch.inference_mode():
        for batch, (x_cpu, y_cpu) in enumerate(windows(validation_ids, 16)):
            x, y = x_cpu.to(device), y_cpu.to(device)
            valid = y != -100; target = y.clamp_min(0).unsqueeze(-1)
            hidden = neural.features(x).float()
            logits = neural.head(hidden) + neural.output_bias
            copy = neural.copy_distribution(hidden, x)
            copy_target = copy.gather(-1, target).squeeze(-1)
            copy_target = torch.where(copy_target > 0, copy_target.log(),
                                      torch.full_like(copy_target, float("-inf")))
            gate_logits = neural.copy_gate(hidden).squeeze(-1)
            count_target = counts.predict_log_probs(x).gather(
                -1, target).squeeze(-1)
            vocabulary_targets = {}
            for temperature in TEMPERATURES:
                for prior in PRIOR_WEIGHTS:
                    vocabulary_targets[(temperature, prior)] = F.log_softmax(
                        logits / temperature + prior * log_prior,
                        dim=-1).gather(-1, target).squeeze(-1)
            candidate_index = 0
            for temperature in TEMPERATURES:
                for prior in PRIOR_WEIGHTS:
                    vocabulary_target = vocabulary_targets[(temperature, prior)]
                    for gate_shift in COPY_GATE_SHIFTS:
                        gate = gate_logits + gate_shift
                        neural_target = torch.logaddexp(
                            F.logsigmoid(-gate) + vocabulary_target,
                            F.logsigmoid(gate) + copy_target)
                        neural_valid = neural_target[valid].unsqueeze(-1)
                        count_valid = count_target[valid].unsqueeze(-1)
                        mixed = torch.logaddexp(
                            neural_valid + torch.log1p(-mixture_tensor)[None, :],
                            count_valid + torch.log(mixture_tensor)[None, :])
                        stop = candidate_index + len(MIXTURE_WEIGHTS)
                        totals[candidate_index:stop] -= mixed.double().sum(0)
                        candidate_index = stop
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    rows = [dict(temperature=key[0], unigram_prior_weight=key[1],
                 copy_gate_shift=key[2], mixture_weight=key[3],
                 nll_nats=value, bpb=value / math.log(2) / len(raw))
            for key, value in zip(grid, totals.cpu().tolist())]
    best = min(rows, key=lambda row: row["bpb"])
    reference = next(row for row in rows
                     if row["temperature"] == 1.10
                     and row["unigram_prior_weight"] == .05
                     and row["copy_gate_shift"] == .1875
                     and row["mixture_weight"] == .075)
    if (targets != 376599 or len(raw) != 1148007
            or abs(reference["bpb"] - REFERENCE_BPB) > 2e-5):
        raise ValueError("Coverage or Stage92 reference mismatch")
    args.run_dir.mkdir(parents=True)
    result = dict(
        protocol=PROTOCOL, split="validation",
        purpose="post_distillation_bounded_scalar_recalibration",
        device=str(device), precision="fp32", neural_sha256=NEURAL_SHA,
        counts_sha256=COUNTS_SHA,
        grid=dict(temperatures=TEMPERATURES,
                  unigram_prior_weights=PRIOR_WEIGHTS,
                  copy_gate_shifts=COPY_GATE_SHIFTS,
                  mixture_weights=MIXTURE_WEIGHTS),
        candidates=rows, reference=reference, best=best,
        bpb_gain=reference["bpb"] - best["bpb"], targets=targets,
        utf8_bytes=len(raw), seconds=time.perf_counter() - started,
        train_unigram_tokens=len(train_ids), new_gradient_targets=0,
        no_test_scoring=True,
        note=("Validation selects four deployment scalars from one fixed local "
              "grid after train-only distillation; no model weights change."),
        source_sha256=sha(Path(__file__)))
    atomic_json_dump(result, args.run_dir / "screen.json")
    print(json.dumps(result | {"candidates": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
