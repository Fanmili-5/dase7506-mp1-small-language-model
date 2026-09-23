"""Continue the full neural expert against the fixed train-only MKN mixture."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from student_mixture_aware import FixedMixtureLM, mixture_log_probs, symmetric_kl
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)

START_SHA = "3be9468b122da4c486726e8bcc59692005d0fc90dcbdd6a59c41b24d42d20b0f"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
STEPS = 3600
BATCH = 32
WEIGHT = .0625
PEAK_LR = 3e-5
AVERAGE_STEPS = (2400, 2700, 3000, 3300, 3600)
SOURCE_FILES = (
    "student_mixture_aware.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram.py", "scripts/train_stage71_mixture_aware.py",
    "train_experiment.py", "evaluate.py", "common.py", "data/manifest.json",
    "data/tokenizer.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    if sha(args.start) != START_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected frozen neural or MKN checkpoint")
    neural_payload = torch.load(args.start, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation")
            != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected Stage67 neural and Stage25 MKN experts")

    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(71017); torch.cuda.manual_seed_all(71017)
    neural, implementation_sha = make_model(
        "student_hybrid_conv_output_bias", neural_payload["config"], device
    )
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"], strict=True)
    counts.load_state_dict(count_payload["model"], strict=True)
    counts.eval()
    for parameter in counts.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        neural.parameters(), lr=PEAK_LR, betas=(.9, .999), weight_decay=.1
    )
    data = load_data(); tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(71017)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    targets_per_step = BATCH * 256
    plan = dict(
        protocol=PROTOCOL, status="training", seed=71017, steps=STEPS,
        primary_targets=STEPS * targets_per_step,
        primary_stochastic_presentations=2 * STEPS * targets_per_step,
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        peak_learning_rate=PEAK_LR, count_weight=WEIGHT, rdrop_alpha=.5,
        comparison="Stage67 neural optimized through fixed Stage68 MKN mixture",
        start_checkpoint_sha256=START_SHA, count_checkpoint_sha256=COUNTS_SHA,
        count_parameters_trainable=0, precision=precision,
        parameters=sum(p.numel() for p in neural.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.; started = time.perf_counter()
    evaluator = FixedMixtureLM(neural, counts, WEIGHT)

    neural.eval()
    before = time.perf_counter()
    initial = score(evaluator, *data["validation"], device, "fp32", 32)
    initial.pop("window_nll_nats")
    validation_seconds += time.perf_counter() - before
    validations.append(dict(step=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)

    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(
            len(tokens) - 257, (BATCH,), generator=rng
        ).to(device)
        sequence = tokens[starts[:, None] + torch.arange(257, device=device)]
        ids, targets = sequence[:, :256], sequence[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        neural.train()
        with torch.no_grad():
            count_logp = counts.predict_log_probs(ids)
        with training_autocast(device, precision):
            first = mixture_log_probs(
                neural.predict_log_probs(ids), count_logp, WEIGHT
            )
            second = mixture_log_probs(
                neural.predict_log_probs(ids), count_logp, WEIGHT
            )
            first_nll = F.nll_loss(first.flatten(0, 1), targets.flatten())
            second_nll = F.nll_loss(second.flatten(0, 1), targets.flatten())
            consistency = symmetric_kl(first, second)
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
            validation = score(evaluator, *data["validation"], device, "fp32", 32)
            validation.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation)
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            if completed in AVERAGE_STEPS:
                atomic_torch_save(
                    checkpoint_payload(
                        neural, "student_hybrid_conv_output_bias",
                        neural_payload["config"], 71017,
                        completed * targets_per_step,
                    ), checkpoints / f"step-{completed:06d}.pt",
                )
            atomic_json_dump(
                dict(completed_steps=completed, history=history,
                     validation_history=validations),
                args.run_dir / "progress.json",
            )
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
