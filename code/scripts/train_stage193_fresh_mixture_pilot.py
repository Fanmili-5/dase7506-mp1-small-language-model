"""Matched train/validation-only Stage193 pilot; never loads the test split."""
from __future__ import annotations

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
from tokenizers import Tokenizer

from common import PROTOCOL, device_metrics, make_model, setup, sha
from scripts.train_stage71_mixture_aware import score_target_mixture
from student_mixture_aware import (build_target_edge_keys,
                                   mixture_target_log_probs, symmetric_kl)
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)


CONFIG = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
CONFIG_SHA = "87aeec0297b7f8b015276cdd51a5e3289de1ba0fef59f6c9845e69967d93fc39"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
COUNT_WEIGHT = 0.0625
SEED = 17
BATCH = 32
PILOT_STEPS = 2400
SCHEDULE_STEPS = 7200
SOURCE_FILES = (
    "student_hybrid_conv_rdrop.py", "student_rdrop_multi_token.py",
    "student_multi_token.py", "student_deep_supervision.py",
    "student_regularized.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "student_mixture_aware.py",
    "student_ngram.py", "scripts/train_stage71_mixture_aware.py",
    "scripts/train_stage193_fresh_mixture_pilot.py", "train_experiment.py",
    "tests/test_stage193_fresh_mixture_pilot.py",
    "docs/STAGE193_FRESH_MIXTURE_AWARE_PILOT_20260928.md",
    "common.py", "configs/stage54_hybrid_conv_rdrop.json",
    "data/manifest.json", "data/tokenizer.json",
    "data/wikitext_train.txt", "data/wikitext_validation.txt",
)


def load_train_validation() -> dict:
    """Verify and tokenize only the two permitted development splits."""
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected benchmark protocol")
    names = ("tokenizer.json", "wikitext_train.txt", "wikitext_validation.txt")
    for name in names:
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError(f"Changed development file: {name}")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    result = {}
    for split in ("train", "validation"):
        raw = (ROOT / "data" / f"wikitext_{split}.txt").read_bytes()
        ids = tokenizer.encode(raw.decode("utf-8")).ids
        result[split] = torch.tensor(ids, dtype=torch.long), len(raw)
    return result


def replace_primary_with_mixture(total: torch.Tensor, logp: torch.Tensor,
                                 count_probability: torch.Tensor,
                                 targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve auxiliary gradients while changing only primary NLL."""
    original = F.nll_loss(logp.flatten(0, 1), targets.flatten())
    mixture = -mixture_target_log_probs(
        logp, count_probability, targets, COUNT_WEIGHT
    ).mean()
    return total - original + mixture, mixture.detach()


def training_loss(model, ids, targets, future_targets, count_probability,
                  arm: str) -> tuple[torch.Tensor, dict]:
    if arm == "control":
        return model.rdrop_training_loss(ids, targets, future_targets)
    if arm != "mixture" or count_probability is None:
        raise ValueError("Mixture arm needs frozen train-only count targets")
    first_total, first_logp, first_parts = model.single_training_pass(
        ids, targets, future_targets
    )
    second_total, second_logp, second_parts = model.single_training_pass(
        ids, targets, future_targets
    )
    first_adjusted, first_mixture = replace_primary_with_mixture(
        first_total, first_logp, count_probability, targets
    )
    second_adjusted, second_mixture = replace_primary_with_mixture(
        second_total, second_logp, count_probability, targets
    )
    with torch.autocast(device_type=ids.device.type, enabled=False):
        consistency = symmetric_kl(first_logp.float(), second_logp.float())
        total = .5 * (first_adjusted + second_adjusted)
        total = total + model.rdrop_alpha * consistency
    parts = {
        "primary": .5 * (first_mixture + second_mixture),
        "deep": .5 * (first_parts["deep"] + second_parts["deep"]),
        "future": .5 * (first_parts["future"] + second_parts["future"]),
        "symmetric_kl": consistency.detach(),
    }
    return total, parts


def one_update(model, counts, edge_keys, tokens, rng, device, precision,
               arm: str, step: int, optimizer) -> dict:
    offsets = tuple(model.future_prediction_offsets)
    span = 256 + max(offsets)
    sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
    starts = sampled.clamp_max(len(tokens) - span)
    extended = tokens[starts[:, None] + torch.arange(span)]
    ids_cpu, targets_cpu = extended[:, :256], extended[:, 1:257]
    future_targets = torch.stack([
        extended[:, offset:offset + 256] for offset in offsets
    ]).to(device)
    count_probability = None
    if arm == "mixture":
        from student_mixture_aware import count_target_probability
        with torch.no_grad():
            count_probability = count_target_probability(
                counts, ids_cpu, targets_cpu, edge_keys
            ).to(device)
    ids, targets = ids_cpu.to(device), targets_cpu.to(device)
    lr = learning_rate(step, SCHEDULE_STEPS, 1e-3, 100, .1, "baseline")
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    model.train()
    with training_autocast(device, precision):
        loss, parts = training_loss(
            model, ids, targets, future_targets, count_probability, arm
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Stage193 loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not torch.isfinite(torch.tensor(grad_norm)):
        raise FloatingPointError("Non-finite Stage193 gradient norm")
    optimizer.step()
    return {
        "step": step + 1, "loss": float(loss.detach()),
        "primary_loss": float(parts["primary"]),
        "deep_loss": float(parts["deep"]),
        "future_loss": float(parts["future"]),
        "symmetric_kl": float(parts["symmetric_kl"]),
        "learning_rate": lr, "grad_norm": grad_norm,
        "clamped_starts": int((starts != sampled).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "mixture"), required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory; no result is overwritten")
    if sha(CONFIG) != CONFIG_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Stage54 config or frozen Stage25 count checkpoint changed")
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    args.run_dir.mkdir(parents=True)
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, implementation_sha = make_model(
        "student_hybrid_conv_rdrop", config, device
    )
    payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if payload.get("protocol") != PROTOCOL or payload.get("implementation") != "student_ngram":
        raise ValueError("Expected frozen train-only Stage25 counts")
    counts, _ = make_model("student_ngram", payload["config"], torch.device("cpu"))
    counts.load_state_dict(payload["model"], strict=True)
    counts.eval()
    for parameter in counts.parameters():
        parameter.requires_grad_(False)
    edge_keys = build_target_edge_keys(counts)
    data = load_train_validation()
    tokens = data["train"][0]
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=.1
    )
    plan = {
        "protocol": PROTOCOL, "stage": "193", "arm": args.arm,
        "preflight_only": args.preflight, "seed": SEED,
        "pilot_steps": PILOT_STEPS, "schedule_steps": SCHEDULE_STEPS,
        "batch": BATCH, "context": 256, "count_weight": COUNT_WEIGHT,
        "count_checkpoint_sha256": COUNTS_SHA,
        "config_sha256": CONFIG_SHA,
        "implementation_sha256": implementation_sha,
        "source_hashes": sources, "no_test_file_opened_by_this_script": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_dump(plan, args.run_dir / "run.json")
    started = time.perf_counter()
    steps = 1 if args.preflight else PILOT_STEPS
    history = []
    total_clamped = 0
    for step in range(steps):
        row = one_update(model, counts, edge_keys, tokens, rng,
                         device, precision, args.arm, step, optimizer)
        total_clamped += row.pop("clamped_starts")
        if args.preflight or (step + 1) % 100 == 0:
            torch.cuda.synchronize(device)
            row["train_seconds"] = time.perf_counter() - started
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Source changed during Stage193 run: {name}")
    if args.preflight:
        peak_allocated = torch.cuda.max_memory_allocated(device)
        memory_ok = peak_allocated <= int(6.5 * 1024**3)
        result = {
            **plan, "status": ("finite_one_update_preflight" if memory_ok
                               else "failed_memory_cap"),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "history": history, "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json_dump(result, args.run_dir / "preflight.json")
        print(json.dumps(result | {"source_hashes": {}}, indent=2), flush=True)
        if not memory_ok:
            raise RuntimeError("Stage193 preflight exceeds 6.5 GiB GPU allocation")
        return
    checkpoint = args.run_dir / "endpoint.pt"
    atomic_torch_save(checkpoint_payload(
        model, "student_hybrid_conv_rdrop", config, SEED,
        PILOT_STEPS * BATCH * 256,
    ), checkpoint)
    model.eval()
    validation = score_target_mixture(
        model, counts, *data["validation"], device, "fp32", edge_keys, BATCH
    )
    result = {
        **plan, "status": "completed_validation_only",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "primary_train_targets": PILOT_STEPS * BATCH * 256,
        "train_seconds": time.perf_counter() - started - validation["seconds"],
        "validation": validation, "checkpoint_sha256": sha(checkpoint),
        "clamped_starts": total_clamped, "history": history,
        **device_metrics(device),
    }
    atomic_json_dump(result, args.run_dir / "metrics.json")
    print(json.dumps(result | {"history": [], "source_hashes": {}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
