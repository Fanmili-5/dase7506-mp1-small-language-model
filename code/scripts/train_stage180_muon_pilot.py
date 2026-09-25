"""Pre-registered 2400-step Muon/AdamW pilot versus matched Stage54 AdamW.

Only hidden block matrices change optimizer. Seed, architecture, data sampling,
loss, context, batch, and 7200-step learning-rate horizon match Stage54.
Continue to a full training run only if validation BPB at step 2400 is at least
0.020 below Stage54's 1.519950369 and training cost remains acceptable.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from muon_pilot import HiddenMatrixMuon, partition_hidden_matrices
from train_experiment import (atomic_json_dump, atomic_torch_save, checkpoint_payload,
                              learning_rate, training_autocast)

STEPS = 2400
HORIZON = 7200
BATCH = 32
CONTEXT = 256
BASELINE_BPB_2400 = 1.519950369
CONTINUE_THRESHOLD = BASELINE_BPB_2400 - 0.020
CONFIG = Path("configs/stage54_hybrid_conv_rdrop.json")
SOURCE_FILES = (
    "muon_pilot.py", "third_party/Muon-LICENSE.txt",
    "student_hybrid_conv_rdrop.py", "student_hybrid_conv_structured.py",
    "student_rdrop_multi_token.py", "student_multi_token.py", "student_deep_supervision.py",
    "student_regularized.py", "student_structured.py", "student.py",
    "scripts/train_stage180_muon_pilot.py", "train_experiment.py", "evaluate.py", "common.py",
    CONFIG.as_posix(), "data/manifest.json", "data/tokenizer.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    args.run_dir.mkdir(parents=True)
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    model, implementation_sha = make_model("student_hybrid_conv_rdrop", config, device)
    hidden, other, group_names = partition_hidden_matrices(model)
    muon = HiddenMatrixMuon(hidden, lr=0.02, momentum=0.95, weight_decay=0.01)
    adam = torch.optim.AdamW(other, lr=0.001, betas=(0.9, 0.999), weight_decay=0.1)
    data = load_data()
    tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(17)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    offsets = tuple(config["future_prediction_offsets"])
    span = CONTEXT + max(offsets)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=17, steps=STEPS,
        learning_rate_horizon=HORIZON, batch=BATCH, context=CONTEXT,
        primary_targets=STEPS * BATCH * CONTEXT,
        primary_stochastic_presentations=2 * STEPS * BATCH * CONTEXT,
        optimizer="Muon(hidden 2D)+AdamW(other)", muon_lr=0.02,
        muon_momentum=0.95, muon_weight_decay=0.01,
        adamw_lr=0.001, adamw_betas=[0.9, 0.999], adamw_weight_decay=0.1,
        muon_source="https://github.com/KellerJordan/Muon/blob/master/muon.py",
        muon_parameter_names=group_names["muon"], adamw_parameter_names=group_names["adamw"],
        prespecified_baseline_step_2400_bpb=BASELINE_BPB_2400,
        prespecified_continue_threshold_bpb=CONTINUE_THRESHOLD,
        decision_rule="continue only if pilot step2400 BPB <= threshold and no unacceptable cost",
        comparison="Stage54 seed17 same architecture, data sampler, loss, and 7200-step LR horizon",
        precision=precision, parameters=sum(p.numel() for p in model.parameters()),
        implementation_sha256=implementation_sha, source_hashes=sources,
        started_utc=datetime.now(timezone.utc).isoformat(), no_test_scoring=True)
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()
    clamped_starts = torch.zeros((), device=device, dtype=torch.int64)
    for step in range(STEPS):
        schedule_factor = learning_rate(step, HORIZON, 1.0, 100, 0.1, "baseline")
        muon.param_groups[0]["lr"] = 0.02 * schedule_factor
        adam.param_groups[0]["lr"] = 0.001 * schedule_factor
        sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=rng).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts.add_((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, targets = extended[:, :CONTEXT], extended[:, 1:CONTEXT + 1]
        future_targets = torch.stack([extended[:, offset:offset + CONTEXT] for offset in offsets])
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        with training_autocast(device, precision):
            loss, parts = model.rdrop_training_loss(ids, targets, future_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        muon.step()
        adam.step()
        completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()), primary_loss=float(parts["primary"]),
                       learning_rate_muon=muon.param_groups[0]["lr"],
                       learning_rate_adamw=adam.param_groups[0]["lr"], grad_norm=grad_norm,
                       primary_targets=completed * BATCH * CONTEXT,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            validation = score(model, *data["validation"], device, "fp32", 32)
            validation.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation)
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations,
                                  clamped_starts=int(clamped_starts)), args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    atomic_torch_save(checkpoint_payload(
        model, "student_hybrid_conv_rdrop", config, 17, STEPS * BATCH * CONTEXT),
        args.run_dir / "checkpoint.pt")
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    measured = validations[-1]["bpb"]
    metrics = dict(
        plan, status="completed_pilot_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, clamped_starts=int(clamped_starts),
        history=history, validation_history=validations,
        final_validation=validations[-1],
        continue_by_bpb=measured <= CONTINUE_THRESHOLD,
        margin_vs_stage54_step2400=measured - BASELINE_BPB_2400,
        checkpoint_sha256=sha(args.run_dir / "checkpoint.pt"), **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
