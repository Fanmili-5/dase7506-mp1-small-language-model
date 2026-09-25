"""Matched Stage54 2,400-step pilot with train-only focused primary loss."""
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

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from student_stage161_frequency_focus import (
    FOCUS_ALPHA, normalized_focused_primary, unseen_medium_frequency_mask)
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate,
                              training_autocast)

SEED = 17
STEPS = 2400
REFERENCE_TOTAL_STEPS = 7200
BATCH = 32
TARGETS = STEPS * BATCH * 256
REFERENCE_BPB = 1.5199503686120217
REFERENCE_TRAINER_SHA = "87ddc44e6a4aee65b39014e24511ebd23dc977475c3279b3e2d94fb242949a4f"
CONFIG = Path("configs/stage54_hybrid_conv_rdrop.json")
SOURCE_FILES = (
    "student_stage161_frequency_focus.py", "student_hybrid_conv_rdrop.py",
    "student_hybrid_conv_structured.py", "student_rdrop_multi_token.py",
    "student_multi_token.py", "student_deep_supervision.py",
    "student_regularized.py", "student_structured.py", "student.py",
    "scripts/train_stage161_frequency_focus_pilot.py",
    "scripts/train_stage54_hybrid_conv_rdrop.py", "train_experiment.py",
    "evaluate.py", "common.py", CONFIG.as_posix(), "data/manifest.json",
    "data/tokenizer.json", "data/wikitext_train.txt",
)


def preflight(config: dict, device: torch.device) -> dict:
    """Prove eval equivalence and the loss mask before spending GPU updates."""
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    original, _ = make_model("student_hybrid_conv_rdrop", config, device)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    focused, _ = make_model("student_stage161_frequency_focus", config, device)
    original.eval(); focused.eval()
    original_state = original.state_dict()
    focused_state = focused.state_dict()
    if original_state.keys() != focused_state.keys() or any(
            not torch.equal(original_state[key], focused_state[key])
            for key in original_state):
        raise ValueError("Stage161 zero-step tensors differ from Stage54")
    ids = torch.randint(2048, (2, 16), device=device)
    with torch.inference_mode():
        first = original.predict_log_probs(ids)
        second = focused.predict_log_probs(ids)
    error = float((first - second).abs().max())
    if error != 0:
        raise ValueError(f"Stage161 zero-step inference differs: {error}")
    toy_ids = torch.tensor([[3, 4, 3]], device=device)
    toy_targets = torch.tensor([[4, 3, 5]], device=device)
    counts = torch.zeros(2048, dtype=torch.long, device=device)
    counts[[3, 4, 5]] = 500
    observed = unseen_medium_frequency_mask(toy_ids, toy_targets, counts)
    if observed.tolist() != [[True, False, True]]:
        raise ValueError("Incorrect within-window training-only focus mask")
    toy_logp = torch.log_softmax(torch.randn(1, 3, 2048, device=device), -1)
    ordinary = -toy_logp.gather(-1, toy_targets[..., None]).mean()
    zero_alpha = normalized_focused_primary(toy_logp, toy_targets,
                                            observed, 0.0)
    if not torch.allclose(ordinary, zero_alpha, rtol=0, atol=1e-6):
        raise ValueError("Alpha-zero focused loss differs from ordinary NLL")
    del original, focused
    torch.cuda.empty_cache()
    return dict(zero_step_state_equal=True, zero_step_max_logp_error=error,
                toy_focus_mask=observed.tolist(),
                alpha_zero_loss_error=float((ordinary - zero_alpha).abs()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage161 run directory")
    if sha(ROOT / "scripts/train_stage54_hybrid_conv_rdrop.py") != REFERENCE_TRAINER_SHA:
        raise ValueError("Matched Stage54 training source changed")
    device, precision = setup("cuda", "bf16", 4)
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    initial_check = preflight(config, device)
    # Restore Stage54's exact initialization and sampler seeds after preflight.
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, implementation_sha = make_model("student_stage161_frequency_focus",
                                           config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    data = load_data()
    train_tokens = data["train"][0]
    frequencies = torch.bincount(train_tokens, minlength=2048)
    model.set_train_frequency(frequencies)
    tokens = train_tokens.to(device)
    sampler = torch.Generator().manual_seed(SEED)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    offsets = tuple(config["future_prediction_offsets"])
    span = 256 + max(offsets)
    args.run_dir.mkdir(parents=True)
    plan = dict(
        protocol=PROTOCOL, status="training", seed=SEED, steps=STEPS,
        reference_total_steps=REFERENCE_TOTAL_STEPS,
        physical_batch=BATCH, effective_batch=BATCH, primary_targets=TARGETS,
        stochastic_primary_presentations=2 * TARGETS,
        objective="Stage54 R-Drop/deep/future plus normalized training-only focus",
        focus_alpha=FOCUS_ALPHA, train_frequency_range=[100, 999],
        prefix_rule="target absent from current independent input prefix 0..t",
        base_control_step2400_bpb=REFERENCE_BPB,
        required_pilot_gain_bpb=.015,
        implementation_sha256=implementation_sha, source_hashes=sources,
        initialization_preflight=initial_check, precision=precision,
        parameters=sum(p.numel() for p in model.parameters()),
        optimizer="AdamW", optimizer_betas=[.9, .999], weight_decay=.1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        training_uses_supplied_train_text_only=True,
        validation_labels_not_used_for_gradients=True, no_test_scoring=True)
    atomic_json_dump(plan, args.run_dir / "run.json")
    started = time.perf_counter(); validation_seconds = 0.0
    history, validations = [], []
    clamped_starts = torch.zeros((), dtype=torch.long, device=device)
    for step in range(STEPS):
        rate = learning_rate(step, REFERENCE_TOTAL_STEPS, 1e-3, 100, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = rate
        sampled = torch.randint(len(tokens) - 257, (BATCH,), generator=sampler).to(device)
        starts = sampled.clamp_max(len(tokens) - span)
        clamped_starts.add_((starts != sampled).sum())
        extended = tokens[starts[:, None] + torch.arange(span, device=device)]
        ids, targets = extended[:, :256], extended[:, 1:257]
        future = torch.stack([extended[:, offset:offset + 256] for offset in offsets])
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with training_autocast(device, precision):
            loss, parts = model.rdrop_training_loss(ids, targets, future)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage161 loss at {step + 1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       primary_loss=float(parts["primary"]),
                       deep_loss=float(parts["deep"]),
                       future_loss=float(parts["future"]),
                       symmetric_kl=float(parts["symmetric_kl"]),
                       focus_fraction=float(parts["focus_fraction"]),
                       learning_rate=rate, grad_norm=grad_norm,
                       primary_targets=completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            result = score(model, *data["validation"], device, "fp32", 32)
            result.pop("window_nll_nats")
            validation_seconds += time.perf_counter() - before
            validations.append(dict(step=completed, **result))
            print(json.dumps({"validation": validations[-1]}), flush=True)
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations,
                                  clamped_starts=int(clamped_starts)),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    checkpoint = args.run_dir / "checkpoint.pt"
    atomic_torch_save(checkpoint_payload(
        model, "student_stage161_frequency_focus", config, SEED, TARGETS),
        checkpoint)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Source changed during Stage161 pilot: {name}")
    endpoint_gain = REFERENCE_BPB - validations[-1]["bpb"]
    metrics = dict(
        plan, status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        train_seconds=time.perf_counter() - started - validation_seconds,
        validation_seconds=validation_seconds,
        clamped_starts=int(clamped_starts), history=history,
        validation_history=validations, final_validation=validations[-1],
        best_validation=min(validations, key=lambda row: row["bpb"]),
        endpoint_gain_vs_matched_control_bpb=endpoint_gain,
        pilot_gate_passed=endpoint_gain >= .015,
        checkpoint_sha256=sha(checkpoint), **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
