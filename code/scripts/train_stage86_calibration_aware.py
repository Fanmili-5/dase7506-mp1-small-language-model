"""Continue Stage71 against the exact frozen Stage79 calibrated mixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from common import (PROTOCOL, autocast, device_metrics, load_data, make_model,
                    setup, sha, windows)
from student_mixture_aware import (build_target_edge_keys,
                                   count_target_probability,
                                   mixture_target_log_probs, symmetric_kl)
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)


START_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
STEPS = 2400
BATCH = 32
COUNT_WEIGHT = .075
TEMPERATURE = 1.10
PRIOR_WEIGHT = .05
GATE_SHIFT = .1875
PEAK_LR = 1e-5
SEED = 86017
EXPECTED_INITIAL_BPB = 1.4030241588936954
AVERAGE_STEPS = (1200, 1500, 1800, 2100, 2400)
SOURCE_FILES = (
    "student_mixture_aware.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram.py", "scripts/train_stage86_calibration_aware.py",
    "train_experiment.py", "evaluate.py", "common.py", "data/manifest.json",
    "data/tokenizer.json",
)


def train_log_prior() -> tuple[torch.Tensor, int]:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_train.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed training data changed")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    text = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(text).ids)
    frequencies = torch.bincount(ids, minlength=2048).double() + .1
    return (frequencies / frequencies.sum()).log().float(), len(ids)


def calibrated_neural_log_probs(
    neural: torch.nn.Module, ids: torch.Tensor, log_prior: torch.Tensor,
) -> torch.Tensor:
    hidden = neural.features(ids)
    with torch.autocast(device_type=ids.device.type, enabled=False):
        hidden = hidden.float()
        vocabulary = F.log_softmax(
            (neural.head(hidden) + neural.output_bias) / TEMPERATURE
            + PRIOR_WEIGHT * log_prior,
            dim=-1,
        )
        copy = neural.copy_distribution(hidden, ids)
        log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
        log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
        gate = neural.copy_gate(hidden) + GATE_SHIFT
        return torch.logaddexp(
            F.logsigmoid(-gate) + vocabulary,
            F.logsigmoid(gate) + log_copy,
        )


def score_target_calibrated_mixture(
    neural, counts, tokens, byte_count, device, precision, edge_keys,
    log_prior, batch_size=32,
):
    neural_mode, count_mode = neural.training, counts.training
    neural.eval(); counts.eval()
    started = time.perf_counter(); nll = 0.; target_count = 0
    with torch.inference_mode():
        for ids, targets in windows(tokens, batch_size):
            count_probability = count_target_probability(
                counts, ids, targets, edge_keys).to(device)
            ids, targets = ids.to(device), targets.to(device)
            with autocast(device, precision):
                neural_logp = calibrated_neural_log_probs(
                    neural, ids, log_prior)
            target_logp = mixture_target_log_probs(
                neural_logp, count_probability, targets, COUNT_WEIGHT)
            valid = targets != -100
            nll -= float(target_logp[valid].double().sum())
            target_count += int(valid.sum())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    neural.train(neural_mode); counts.train(count_mode)
    return dict(
        bpb=nll / math.log(2.) / byte_count,
        token_ppl=math.exp(nll / target_count), nll_nats=nll,
        targets=target_count, utf8_bytes=byte_count,
        seconds=time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    if sha(args.start) != START_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected Stage71 neural or MKN checkpoint")
    neural_payload = torch.load(args.start, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected Stage71 neural and Stage25 MKN experts")

    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    neural, implementation_sha = make_model(
        neural_payload["implementation"], neural_payload["config"], device)
    counts, _ = make_model(
        count_payload["implementation"], count_payload["config"],
        torch.device("cpu"))
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True); counts.eval()
    edge_keys = build_target_edge_keys(counts)
    for parameter in counts.parameters():
        parameter.requires_grad_(False)
    log_prior, train_unigram_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    optimizer = torch.optim.AdamW(
        neural.parameters(), lr=PEAK_LR, betas=(.9, .999), weight_decay=.1)
    data = load_data(); tokens = data["train"][0]
    rng = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    targets_per_step = BATCH * 256
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
        primary_targets=STEPS * targets_per_step,
        primary_stochastic_presentations=2 * STEPS * targets_per_step,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR, rdrop_alpha=.5,
        temperature=TEMPERATURE, train_unigram_prior_weight=PRIOR_WEIGHT,
        copy_gate_shift=GATE_SHIFT, count_weight=COUNT_WEIGHT,
        train_unigram_tokens=train_unigram_tokens,
        comparison="Stage71 neural optimized through frozen Stage79 calibration",
        start_checkpoint_sha256=START_SHA, count_checkpoint_sha256=COUNTS_SHA,
        count_parameters_trainable=0, count_lookup_device="cpu",
        neural_training_device=str(device), precision=precision,
        parameters=sum(p.numel() for p in neural.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.; started = time.perf_counter()
    neural.eval(); before = time.perf_counter()
    initial = score_target_calibrated_mixture(
        neural, counts, *data["validation"], device, "fp32", edge_keys,
        log_prior, 32)
    if abs(initial["bpb"] - EXPECTED_INITIAL_BPB) > 2e-6:
        raise ValueError("Calibrated target scorer disagrees with Stage79")
    validation_seconds += time.perf_counter() - before
    validations.append(dict(step=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)

    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids, targets = sequence[:, :256], sequence[:, 1:]
        with torch.no_grad():
            count_probability = count_target_probability(
                counts, ids, targets, edge_keys).to(device)
        ids, targets = ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True); neural.train()
        with training_autocast(device, precision):
            first_neural = calibrated_neural_log_probs(neural, ids, log_prior)
            second_neural = calibrated_neural_log_probs(neural, ids, log_prior)
            first_target = mixture_target_log_probs(
                first_neural, count_probability, targets, COUNT_WEIGHT)
            second_target = mixture_target_log_probs(
                second_neural, count_probability, targets, COUNT_WEIGHT)
            first_nll = -first_target.mean(); second_nll = -second_target.mean()
            consistency = symmetric_kl(first_neural, second_neural)
            loss = .5 * (first_nll + second_nll) + .5 * consistency
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(neural.parameters(), 1.0))
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(
                step=completed, loss=float(loss.detach()),
                mixture_nll=float(.5 * (first_nll + second_nll).detach()),
                symmetric_kl=float(consistency.detach()), learning_rate=lr,
                grad_norm=grad_norm,
                primary_targets=completed * targets_per_step,
                train_seconds=time.perf_counter() - started - validation_seconds,
            )
            history.append(row); print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            neural.eval(); before = time.perf_counter()
            validation = score_target_calibrated_mixture(
                neural, counts, *data["validation"], device, "fp32", edge_keys,
                log_prior, 32)
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation)
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            if completed in AVERAGE_STEPS:
                atomic_torch_save(
                    checkpoint_payload(
                        neural, neural_payload["implementation"],
                        neural_payload["config"], SEED,
                        completed * targets_per_step),
                    checkpoints / f"step-{completed:06d}.pt")
            atomic_json_dump(
                dict(completed_steps=completed, history=history,
                     validation_history=validations),
                args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(), history=history,
        validation_history=validations, final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, **device_metrics(device),
    )
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
