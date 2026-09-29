"""Continue Stage63 with training-only byte composition and primary emphasis."""
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
from train_experiment import (
    atomic_json_dump,
    atomic_torch_save,
    checkpoint_payload,
    learning_rate,
    training_autocast,
)

START_SHA = "671bb26dfe611e7d6edb76b1910c52e1f67b131ae8f77a11836b6a4ebfd46a11"
STEPS = 3600
PRIOR_STEPS = 20400
BATCH = 32
TARGETS = STEPS * BATCH * 256
AVERAGE_STEPS = (2400, 2700, 3000, 3300, 3600)
BASE_PEAK_LR = 3e-5
BYTE_PEAK_LR = 3e-4
AUX_WEIGHT = .05
CONFIG = Path("configs/stage65_hybrid_conv_byte_rdrop.json")
SOURCE_FILES = (
    "student_hybrid_conv_byte_rdrop.py", "student_hybrid_conv_rdrop.py",
    "student_hybrid_conv_structured.py", "student_byte_composed.py",
    "student_rdrop_multi_token.py", "student_multi_token.py",
    "student_deep_supervision.py", "student_regularized.py",
    "student_structured.py", "student.py",
    "scripts/train_stage65_hybrid_conv_byte_rdrop.py", "train_experiment.py",
    "evaluate.py", "common.py", CONFIG.as_posix(),
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
        raise ValueError("Unexpected Stage63 averaged training checkpoint")
    start_payload = torch.load(args.start, map_location="cpu", weights_only=True)
    if (start_payload.get("protocol") != PROTOCOL
            or start_payload.get("implementation") != "student_hybrid_conv_rdrop"):
        raise ValueError("Expected the Stage63 hybrid-conv R-Drop training average")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    parent_config = dict(config); parent_config.pop("byte_features")
    if start_payload["config"] != parent_config:
        raise ValueError("Stage63 configuration changed")

    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    model, implementation_sha = make_model(
        "student_hybrid_conv_byte_rdrop", config, device
    )
    incompatible = model.load_state_dict(start_payload["model"], strict=False)
    if incompatible.missing_keys != ["byte_projection"] or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected Stage63 load result: {incompatible}")
    model.deep_supervision_weight = AUX_WEIGHT
    model.future_prediction_weight = AUX_WEIGHT

    byte_parameters = [model.byte_projection]
    base_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name != "byte_projection"
    ]
    torch.manual_seed(65017); torch.cuda.manual_seed_all(65017)
    optimizer = torch.optim.AdamW(
        [
            dict(params=base_parameters, lr=BASE_PEAK_LR, weight_decay=.1),
            dict(params=byte_parameters, lr=BYTE_PEAK_LR, weight_decay=0.),
        ],
        betas=(.9, .999),
    )
    data = load_data(); tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(17)
    for _ in range(PRIOR_STEPS):
        torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    offsets = tuple(config["future_prediction_offsets"]); span = 256 + max(offsets)
    plan = dict(
        protocol=PROTOCOL, status="training", start_checkpoint_sha256=START_SHA,
        start_stage="Stage63 fixed five-checkpoint training average",
        seed=17, dropout_rng_seed=65017, prior_sampling_steps=PRIOR_STEPS,
        continuation_steps=STEPS, additional_primary_targets=TARGETS,
        primary_stochastic_presentations=2 * TARGETS,
        deep_supervision_label_presentations=2 * TARGETS * len(
            config["deep_supervision_layers"]),
        future_label_presentations=2 * TARGETS * len(offsets),
        future_offsets=list(offsets), rdrop_alpha=config["rdrop_alpha"],
        stochastic_forwards_per_window=2,
        deep_supervision_weight=AUX_WEIGHT,
        future_prediction_weight=AUX_WEIGHT,
        byte_feature_kind=config["byte_features"],
        byte_projection_initialization="exact_zero",
        optimizer="fresh AdamW with base/byte parameter groups",
        optimizer_betas=[.9, .999], base_weight_decay=.1, byte_weight_decay=0.,
        base_peak_learning_rate=BASE_PEAK_LR,
        byte_peak_learning_rate=BYTE_PEAK_LR,
        minimum_learning_rate_fraction=.1, warmup_steps=50,
        comparison="Stage63 plus zero-initialized training-only byte composition",
        precision=precision, parameters=sum(p.numel() for p in model.parameters()),
        deployed_parameter_increment=0, implementation_sha256=implementation_sha,
        source_hashes=sources, started_utc=datetime.now(timezone.utc).isoformat(),
        no_test_scoring=True,
    )
    atomic_json_dump(plan, args.run_dir / "run.json")
    history, validations = [], []
    started = time.perf_counter(); before = time.perf_counter()
    initial = score(model, *data["validation"], device, "fp32", 32)
    validation_seconds = time.perf_counter() - before
    initial.pop("window_nll_nats")
    validations.append(dict(step=0, **initial))
    print(json.dumps({"validation": validations[-1]}), flush=True)
    clamped_starts = torch.zeros((), device=device, dtype=torch.int64)

    for step in range(STEPS):
        base_lr = learning_rate(
            step, STEPS, BASE_PEAK_LR, 50, .1, "warmup_cosine"
        )
        byte_lr = base_lr * (BYTE_PEAK_LR / BASE_PEAK_LR)
        optimizer.param_groups[0]["lr"] = base_lr
        optimizer.param_groups[1]["lr"] = byte_lr
        sampled = torch.randint(
            len(tokens) - 257, (BATCH,), generator=rng
        ).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts.add_((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, primary_targets = extended[:, :256], extended[:, 1:257]
        future_targets = torch.stack([
            extended[:, offset:offset + 256] for offset in offsets
        ])
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, precision):
            loss, parts = model.rdrop_training_loss(
                ids, primary_targets, future_targets
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        byte_grad_norm = float(model.byte_projection.grad.float().norm())
        optimizer.step(); completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(
                step=completed, loss=float(loss.detach()),
                primary_loss=float(parts["primary"]), deep_loss=float(parts["deep"]),
                future_loss=float(parts["future"]),
                symmetric_kl=float(parts["symmetric_kl"]),
                base_learning_rate=base_lr, byte_learning_rate=byte_lr,
                grad_norm=grad_norm, byte_grad_norm=byte_grad_norm,
                additional_primary_targets=completed * BATCH * 256,
                train_seconds=time.perf_counter() - started - validation_seconds,
            )
            history.append(row); print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            validation = score(model, *data["validation"], device, "fp32", 32)
            validation.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            row = dict(step=completed, **validation)
            validations.append(row); print(json.dumps({"validation": row}), flush=True)
            if completed in AVERAGE_STEPS:
                atomic_torch_save(
                    checkpoint_payload(
                        model, "student_hybrid_conv_byte_rdrop", config, 17,
                        (PRIOR_STEPS + completed) * BATCH * 256,
                    ),
                    checkpoints / f"step-{completed:06d}.pt",
                )
            atomic_json_dump(
                dict(completed_steps=completed, history=history,
                     validation_history=validations,
                     clamped_starts=int(clamped_starts)),
                args.run_dir / "progress.json",
            )
    torch.cuda.synchronize(device)
    atomic_torch_save(
        checkpoint_payload(
            model, "student_hybrid_conv_byte_rdrop", config, 17,
            (PRIOR_STEPS + STEPS) * BATCH * 256,
        ),
        args.run_dir / "checkpoint.pt",
    )
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during training: " + name)
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        additional_train_targets=TARGETS,
        cumulative_nominal_train_targets=(PRIOR_STEPS + STEPS) * BATCH * 256,
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds,
        clamped_starts=int(clamped_starts), history=history,
        validation_history=validations, final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        checkpoint_sha256=sha(args.run_dir / "checkpoint.pt"),
        **device_metrics(device),
    )
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
