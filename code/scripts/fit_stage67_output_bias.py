"""Fit only a vocabulary intercept on supplied training windows."""
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

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha, windows
from evaluate import score
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload

START_SHA = "f9c8b41efea91d5594b25760c91652edc071a161983b12b22cc738988a7a9bdc"
EPOCHS = 5
BATCH = 64
PEAK_LR = .01
CONFIG = Path("configs/stage67_hybrid_conv_output_bias.json")
SOURCE_FILES = (
    "student_hybrid_conv_output_bias.py", "student_hybrid_conv_structured.py",
    "student_structured.py", "student.py", "scripts/fit_stage67_output_bias.py",
    "train_experiment.py", "evaluate.py", "common.py", CONFIG.as_posix(),
    "data/manifest.json", "data/tokenizer.json",
)


def main():
    global START_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fresh-run", action="store_true",
                        help="Accept newly trained inputs; keep protocol checks and record actual hashes.")
    args = parser.parse_args()
    if args.fresh_run:
        START_SHA = sha(args.start)
    if args.run_dir.exists():
        parser.error("Choose a new run directory")
    if sha(args.start) != START_SHA:
        raise ValueError("Unexpected Stage65 neural checkpoint")
    payload = torch.load(args.start, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_hybrid_conv_structured"):
        raise ValueError("Expected the Stage65 deployed hybrid-conv neural")
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    source_config = dict(config); source_config.pop("output_bias")
    if payload["config"] != source_config:
        raise ValueError("Stage65 deployed configuration changed")

    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    model, implementation_sha = make_model(
        "student_hybrid_conv_output_bias", config, device
    )
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.missing_keys != ["output_bias"] or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected Stage65 load result: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.output_bias.requires_grad_(True)
    model.eval()
    optimizer = torch.optim.Adam([model.output_bias], lr=PEAK_LR)
    data = load_data(); train_tokens = data["train"][0]
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    train_targets_per_epoch = len(train_tokens) - 1
    plan = dict(
        protocol=PROTOCOL, status="training", start_checkpoint_sha256=START_SHA,
        start_stage="Stage65 materialized neural average", epochs=EPOCHS,
        batch_size=BATCH, optimizer="Adam", learning_rate=PEAK_LR,
        trained_parameters=model.output_bias.numel(),
        frozen_parameters=sum(p.numel() for n, p in model.named_parameters()
                              if n != "output_bias"),
        targets_per_epoch=train_targets_per_epoch,
        total_train_target_presentations=EPOCHS * train_targets_per_epoch,
        windowing="deterministic independent non-overlapping 256-token windows",
        output_bias_initialization="exact_zero", mean_zero_projection=True,
        comparison="Stage65 plus train-only vocabulary intercept fitting",
        precision=precision, implementation_sha256=implementation_sha,
        source_hashes=sources, started_utc=datetime.now(timezone.utc).isoformat(),
        no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    validations = []
    started = time.perf_counter(); before = time.perf_counter()
    initial = score(model, *data["validation"], device, "fp32", 32)
    validation_seconds = time.perf_counter() - before
    initial.pop("window_nll_nats")
    validations.append(dict(epoch=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)
    history = []
    observed_targets = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_nll = 0.; epoch_targets = 0
        for ids, targets in windows(train_tokens, BATCH):
            ids, targets = ids.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logp = model.predict_log_probs(ids)
                loss = F.nll_loss(
                    logp.flatten(0, 1), targets.flatten(), ignore_index=-100
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in epoch {epoch}")
            loss.backward()
            if (model.output_bias.grad is None
                    or not torch.isfinite(model.output_bias.grad).all()):
                raise FloatingPointError("Invalid output-bias gradient")
            optimizer.step()
            with torch.no_grad():
                model.output_bias.sub_(model.output_bias.mean())
            count = int((targets != -100).sum())
            epoch_nll += float(loss.detach()) * count
            epoch_targets += count
        if epoch_targets != train_targets_per_epoch:
            raise ValueError("Training-window target coverage changed")
        observed_targets += epoch_targets
        row = dict(
            epoch=epoch, train_nll_nats=epoch_nll,
            train_token_ppl=math.exp(epoch_nll / epoch_targets),
            output_bias_l2=float(model.output_bias.detach().norm()),
            output_bias_max_abs=float(model.output_bias.detach().abs().max()),
        )
        history.append(row); print(json.dumps(row), flush=True)
        before = time.perf_counter()
        validation = score(model, *data["validation"], device, "fp32", 32)
        validation.pop("window_nll_nats")
        validation_seconds += time.perf_counter() - before
        validation_row = dict(epoch=epoch, **validation)
        validations.append(validation_row)
        print(json.dumps({"validation": validation_row}), flush=True)
        atomic_torch_save(
            checkpoint_payload(
                model, "student_hybrid_conv_output_bias", config, 17,
                observed_targets,
            ),
            checkpoints / f"epoch-{epoch:02d}.pt",
        )
        atomic_json_dump(
            dict(completed_epochs=epoch, history=history,
                 validation_history=validations,
                 observed_train_targets=observed_targets),
            args.run_dir / "progress.json",
        )
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during fitting: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(), history=history,
        validation_history=validations,
        final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        observed_train_targets=observed_targets,
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds, **device_metrics(device),
    )
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
