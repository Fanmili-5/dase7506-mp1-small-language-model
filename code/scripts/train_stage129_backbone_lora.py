"""Distill frozen heterogeneous teacher into mergeable backbone LoRA only."""
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
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.screen_stage112_block_ablation import EXPECTED_BYTES, EXPECTED_TARGETS, score
from scripts.train_stage86_calibration_aware import calibrated_neural_log_probs, train_log_prior
from scripts.train_stage92_heterogeneous_distillation import PRIMARY_SHA, ALTERNATE_SHA
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import (atomic_json_dump, atomic_torch_save,
                              checkpoint_payload, learning_rate, training_autocast)


RANK, SEED, BATCH, STEPS = 8, 129017, 24, 1800
PEAK_LR, WEIGHT_DECAY = .002, .01
SAVE_STEPS = (900, 1200, 1500, 1800)
REFERENCE_BPB = 1.4017076627486786
SOURCE_FILES = (
    "scripts/train_stage129_backbone_lora.py", "student_hybrid_conv_output_bias.py",
    "student_hybrid_conv_structured.py", "student_structured.py", "student.py",
    "student_ngram.py", "student_mixture_aware.py",
    "scripts/analyze_stage98_distilled_gate.py",
    "scripts/screen_stage112_block_ablation.py",
    "scripts/train_stage86_calibration_aware.py",
    "scripts/train_stage92_heterogeneous_distillation.py",
    "train_experiment.py", "common.py", "evaluate.py", "data/manifest.json",
    "data/tokenizer.json",
)


class LowRankDelta(nn.Module):
    def __init__(self, output: int, input_: int, rank: int = RANK):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, input_))
        self.b = nn.Parameter(torch.zeros(output, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0  # alpha=rank, hence alpha/rank=1

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + self.scale * (self.b @ self.a)


def internal_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    found = []
    for index, block in enumerate(model.blocks):
        for name in ("qkv", "input"):
            layer = getattr(block, name, None)
            if isinstance(layer, nn.Linear):
                found.append((f"blocks.{index}.{name}", layer))
        projection = getattr(block, "proj")
        found.append((f"blocks.{index}.proj", projection))
        found.append((f"blocks.{index}.mlp.input", block.mlp.input))
        found.append((f"blocks.{index}.mlp.output", block.mlp.output))
    if len(found) != 32 or any(not isinstance(layer, nn.Linear) for _, layer in found):
        raise ValueError("Stage129 expected exactly 32 backbone linears")
    return found


def attach_lora(model: nn.Module) -> tuple[list[str], list[nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = internal_linears(model)
    for _, layer in targets:
        parametrize.register_parametrization(
            layer, "weight", LowRankDelta(
                layer.out_features, layer.in_features).to(layer.weight.device))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable) != 64:
        raise ValueError("Expected A/B parameters in 32 internal linears")
    return [name for name, _ in targets], trainable


def merged_model(model: nn.Module):
    """Materialize effective weights without mutating PyTorch's shared wrapper class."""
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        merged, _ = make_model("student_hybrid_conv_output_bias", model.config, device)
    source = model.state_dict()
    destination = merged.state_dict()
    adapted_linears = dict(internal_linears(model))
    with torch.no_grad():
        for key in destination:
            if key in source:
                destination[key] = source[key].detach().clone()
            elif key.endswith(".weight") and key[:-7] in adapted_linears:
                destination[key] = adapted_linears[key[:-7]].weight.detach().clone()
            else:
                raise ValueError(f"Cannot materialize LoRA state: {key}")
    merged.load_state_dict(destination, strict=True)
    merged.eval()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage129 directory")
    expected = ((args.student, NEURAL_SHA), (args.primary, PRIMARY_SHA),
                (args.alternate, ALTERNATE_SHA), (args.counts, BASE_SHA))
    if any(sha(path) != digest for path, digest in expected):
        raise ValueError("Stage129 frozen ancestry changed")
    payloads = [torch.load(path, map_location="cpu", weights_only=True)
                for path, _ in expected]
    student_payload, primary_payload, alternate_payload, count_payload = payloads
    if (any(payload.get("protocol") != PROTOCOL for payload in payloads)
            or student_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or primary_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or alternate_payload.get("implementation") != "student_hybrid_conv_structured"
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected Stage129 checkpoint implementation")
    device, precision = setup("cuda", "bf16", 4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    student, _ = make_model(student_payload["implementation"],
                            student_payload["config"], device)
    primary, _ = make_model(primary_payload["implementation"],
                            primary_payload["config"], device)
    alternate, _ = make_model(alternate_payload["implementation"],
                              alternate_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], torch.device("cpu"))
    for model, payload in zip((student, primary, alternate, counts), payloads):
        model.load_state_dict(payload["model"], strict=True)
    student.eval(); primary.eval(); alternate.eval(); counts.eval()
    for model in (primary, alternate, counts):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    log_prior, prior_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    data = load_data()
    tokens = data["train"][0]
    validation, raw_bytes = data["validation"]
    if raw_bytes != EXPECTED_BYTES:
        raise ValueError("Validation byte count changed")
    edge_keys = build_target_edge_keys(counts)
    with torch.inference_mode():
        validation_batches = [
            (ids, labels, count_target_probability(counts, ids, labels, edge_keys))
            for ids, labels in windows(validation, 32)]

    smoke_ids = (torch.arange(512, device=device).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        pre_attach = distilled_log_probs(student, smoke_ids, log_prior)
    names, trainable = attach_lora(student)
    with torch.inference_mode():
        attached = distilled_log_probs(student, smoke_ids, log_prior)
    initial_error = float((pre_attach.exp() - attached.exp()).abs().max())
    if initial_error > 3e-6:
        raise ValueError(f"LoRA zero-delta initialization mismatch: {initial_error}")
    merged_initial = merged_model(student)
    with torch.inference_mode():
        initial_merged = distilled_log_probs(merged_initial, smoke_ids, log_prior)
    initial_merge_error = float((attached.exp() - initial_merged.exp()).abs().max())
    if initial_merge_error > 3e-5:
        raise ValueError(f"LoRA merge mismatch: {initial_merge_error}")
    del merged_initial
    optimizer = torch.optim.AdamW(trainable, lr=PEAK_LR, betas=(.9, .999),
                                  weight_decay=WEIGHT_DECAY)
    rng = torch.Generator().manual_seed(SEED)
    sources = {file: sha(ROOT / file) for file in SOURCE_FILES}
    initial = score(student, validation_batches, log_prior, raw_bytes)
    if (initial["targets"] != EXPECTED_TARGETS
            or abs(initial["bpb"] - REFERENCE_BPB) > 2e-5):
        raise ValueError(f"Stage92 start failed reproduction: {initial}")

    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    checkpoints.mkdir()
    plan = dict(
        protocol=PROTOCOL, status="training", mechanism="mergeable_backbone_lora",
        seed=SEED, rank=RANK, steps=STEPS, batch_size=BATCH,
        new_training_targets=STEPS * BATCH * 256,
        previous_training_targets=int(student_payload.get("train_tokens", 0)),
        peak_learning_rate=PEAK_LR, weight_decay=WEIGHT_DECAY,
        trained_parameters=sum(p.numel() for p in trainable),
        adapted_linears=names, initial_probability_error=initial_error,
        initial_merge_probability_error=initial_merge_error,
        fixed_teacher_weights=[.55, .45], teacher_loss_weight=.75,
        hard_loss_weight=.25, selected_average_steps=SAVE_STEPS,
        source_hashes=sources,
        source_checkpoint_hashes=dict(student=NEURAL_SHA, primary=PRIMARY_SHA,
                                      alternate=ALTERNATE_SHA, counts=BASE_SHA),
        train_unigram_tokens=prior_tokens,
        validation_labels_not_used_for_gradients=True,
        teacher_uses_training_prefixes_only=True,
        no_test_scoring=True, precision=precision, device=str(device),
        started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json_dump(plan, args.run_dir / "run.json")
    print(json.dumps({"validation": dict(step=0, **initial)}), flush=True)
    history = []
    validations = [dict(step=0, **initial)]
    started = time.perf_counter()
    validation_seconds = 0.0
    for step in range(STEPS):
        completed = step + 1
        lr = learning_rate(step, STEPS, PEAK_LR, 50, .1, "baseline")
        for group in optimizer.param_groups:
            group["lr"] = lr
        starts = torch.randint(len(tokens) - 257, (BATCH,), generator=rng)
        sequence = tokens[starts[:, None] + torch.arange(257)]
        ids, labels = sequence[:, :256].to(device), sequence[:, 1:].to(device)
        with torch.no_grad(), training_autocast(device, precision):
            teacher_primary = calibrated_neural_log_probs(primary, ids, log_prior)
            teacher_alternate = alternate.predict_log_probs(ids)
            teacher_logp = torch.logaddexp(
                teacher_primary + math.log(.55),
                teacher_alternate + math.log(.45))
            teacher_probability = teacher_logp.exp()
        optimizer.zero_grad(set_to_none=True)
        student.train()
        with training_autocast(device, precision):
            student_logp = distilled_log_probs(student, ids, log_prior)
            distillation = -(teacher_probability * student_logp).sum(-1).mean()
            hard = F.nll_loss(student_logp.flatten(0, 1), labels.flatten())
            loss = .75 * distillation + .25 * hard
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite Stage129 loss at step {completed}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if completed == 1 and (not math.isfinite(grad_norm) or grad_norm <= 0):
            raise ValueError("Stage129 adapter gradients failed first-step preflight")
        optimizer.step()
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                       teacher_cross_entropy=float(distillation.detach()),
                       hard_nll=float(hard.detach()), learning_rate=lr,
                       grad_norm=grad_norm,
                       new_training_targets=completed * BATCH * 256,
                       train_seconds=time.perf_counter() - started - validation_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            student.eval()
            before = time.perf_counter()
            measured = score(student, validation_batches, log_prior, raw_bytes)
            validation_seconds += time.perf_counter() - before
            if measured["targets"] != EXPECTED_TARGETS:
                raise ValueError("Stage129 validation target coverage mismatch")
            row = dict(step=completed, **measured)
            validations.append(row)
            print(json.dumps({"validation": row}), flush=True)
            if completed in SAVE_STEPS:
                merged = merged_model(student)
                with torch.inference_mode():
                    adapted = distilled_log_probs(student, smoke_ids, log_prior)
                    merged_logp = distilled_log_probs(merged, smoke_ids, log_prior)
                merge_error = float((adapted.exp() - merged_logp.exp()).abs().max())
                if merge_error > 3e-5:
                    raise ValueError(f"Stage129 trained merge mismatch: {merge_error}")
                payload = checkpoint_payload(
                    merged, student_payload["implementation"],
                    student_payload["config"], SEED,
                    int(student_payload.get("train_tokens", 0))
                    + completed * BATCH * 256)
                payload["merged_adapter_ancestry"] = dict(
                    start_sha256=NEURAL_SHA, rank=RANK, step=completed,
                    merge_max_probability_error=merge_error,
                    no_test_scoring=True)
                atomic_torch_save(payload, checkpoints / f"step-{completed:06d}.pt")
                del merged
            atomic_json_dump(dict(completed_steps=completed, history=history,
                                  validation_history=validations),
                             args.run_dir / "progress.json")
    torch.cuda.synchronize(device)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed during Stage129: " + name)
    metrics = dict(plan, status="completed_training_validation_only",
                   completed_utc=datetime.now(timezone.utc).isoformat(),
                   history=history, validation_history=validations,
                   final_validation=validations[-1],
                   best_validation=min(validations, key=lambda row: row["bpb"]),
                   train_seconds=time.perf_counter() - started - validation_seconds,
                   validation_seconds=validation_seconds, **device_metrics(device))
    atomic_json_dump(metrics, args.run_dir / "metrics.json")
    print(json.dumps(metrics | {"history": [], "validation_history": []},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
