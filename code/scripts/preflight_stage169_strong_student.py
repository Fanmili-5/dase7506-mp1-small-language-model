"""Train-only differentiability/resource preflight for Stage169."""
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

from common import PROTOCOL, load_data, make_model, setup, sha
from scripts.preflight_stage157_teacher import STAGE105_SHA, STAGE155_SHA, load_checkpoint
from train_experiment import atomic_json_dump, training_autocast

SEED = 169017
BATCH = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage105", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new preflight output")
    device, precision = setup("cuda", "bf16", 4)
    if device.type != "cuda":
        raise ValueError("Stage169 feasibility must be measured on the Windows GPU")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    old, old_payload, old_module_sha = load_checkpoint(args.stage105, STAGE105_SHA, device)
    new, new_payload, new_module_sha = load_checkpoint(args.stage155, STAGE155_SHA, device)
    if (old_payload["implementation"] != "student_stage105_gated_singlepass"
            or new_payload["implementation"] != "student_hybrid_conv_rdrop"):
        raise ValueError("Unexpected teacher architectures")
    student, student_module_sha = make_model("student_stage103_gated", old_payload["config"], device)
    student.load_state_dict(old_payload["model"], strict=True)
    for teacher in (old, new):
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    train = load_data()["train"][0]
    sampler = torch.Generator().manual_seed(SEED)
    starts = torch.randint(len(train) - 257, (BATCH,), generator=sampler)
    sequence = train[starts[:, None] + torch.arange(257)]
    ids, targets = sequence[:, :256].to(device), sequence[:, 1:].to(device)
    with torch.no_grad():
        old_logp = old.predict_log_probs(ids).float()
        new_logp = new.predict_log_probs(ids).float()
        teacher_logp = torch.logaddexp(old_logp, new_logp) - math.log(2)
        teacher_probability = teacher_logp.exp()
        student.eval()
        student_initial = student.predict_log_probs(ids).float()
        parity = float((student_initial - old_logp).abs().max())
        normalization = float(teacher_logp.logsumexp(-1).abs().max())
    student.train()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with training_autocast(device, precision):
        student_logp = student.predict_log_probs(ids).float()
        distill = -(teacher_probability * student_logp).sum(-1).mean()
        hard = F.nll_loss(student_logp.reshape(-1, 2048), targets.reshape(-1))
        loss = .75 * distill + .25 * hard
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Stage169 loss")
    loss.backward()
    torch.cuda.synchronize(device)
    gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                           for p in student.parameters())
    nonzero_gradients = sum(p.grad is not None and bool((p.grad != 0).any())
                            for p in student.parameters())
    result = dict(
        protocol=PROTOCOL, purpose="stage169_train_only_feasibility",
        stage105_sha256=STAGE105_SHA, stage155_sha256=STAGE155_SHA,
        old_module_sha256=old_module_sha, new_module_sha256=new_module_sha,
        student_module_sha256=student_module_sha, source_sha256=sha(Path(__file__)),
        physical_batch=BATCH, precision=precision, source_split="train",
        max_student_vs_stage105_logp_error=parity,
        max_teacher_log_normalization_error=normalization,
        loss=float(loss.detach()), distill_loss=float(distill.detach()),
        hard_loss=float(hard.detach()), gradients_finite=gradients_finite,
        nonzero_gradient_tensors=nonzero_gradients,
        one_step_wall_seconds=time.perf_counter() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        no_test_scoring=True,
    )
    result["pilot_authorized"] = (
        parity <= 3e-4 and normalization <= 1e-3 and gradients_finite
        and nonzero_gradients > 0 and result["one_step_wall_seconds"] <= 6
        and result["peak_cuda_allocated_bytes"] <= 6_500_000_000)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
