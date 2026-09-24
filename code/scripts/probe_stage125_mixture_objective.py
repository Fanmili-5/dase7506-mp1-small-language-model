"""One-batch train-prefix feasibility probe for Stage125 final-mixture CE."""
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
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.train_stage86_calibration_aware import calibrated_neural_log_probs, train_log_prior
from scripts.train_stage92_heterogeneous_distillation import PRIMARY_SHA, ALTERNATE_SHA
from train_experiment import training_autocast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage125 probe output")
    expected = ((args.student, NEURAL_SHA), (args.primary, PRIMARY_SHA),
                (args.alternate, ALTERNATE_SHA), (args.counts, BASE_SHA))
    if any(sha(path) != digest for path, digest in expected):
        raise ValueError("Unexpected frozen Stage125 ancestry")
    payloads = [torch.load(path, map_location="cpu", weights_only=True)
                for path, _ in expected]
    if any(payload.get("protocol") != PROTOCOL for payload in payloads):
        raise ValueError("Unexpected protocol")
    device, precision = setup("cuda", "bf16", 4)
    student, _ = make_model(payloads[0]["implementation"], payloads[0]["config"], device)
    primary, _ = make_model(payloads[1]["implementation"], payloads[1]["config"], device)
    alternate, _ = make_model(payloads[2]["implementation"], payloads[2]["config"], device)
    counts, _ = make_model(payloads[3]["implementation"], payloads[3]["config"],
                           torch.device("cpu"))
    for model, payload in zip((student, primary, alternate, counts), payloads):
        model.load_state_dict(payload["model"], strict=True)
    student.train(); primary.eval(); alternate.eval(); counts.eval()
    for model in (primary, alternate, counts):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    log_prior, train_tokens = train_log_prior()
    log_prior = log_prior.to(device)
    tokens = load_data()["train"][0]
    starts = torch.randint(len(tokens) - 257, (24,), generator=torch.Generator().manual_seed(125017))
    sequence = tokens[starts[:, None] + torch.arange(257)]
    ids_cpu, labels_cpu = sequence[:, :256], sequence[:, 1:]
    started = time.perf_counter()
    with torch.no_grad():
        count_logp = counts.predict_log_probs(ids_cpu).to(device)
    count_seconds = time.perf_counter() - started
    ids, labels = ids_cpu.to(device), labels_cpu.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad(), training_autocast(device, precision):
        primary_logp = calibrated_neural_log_probs(primary, ids, log_prior)
        alternate_logp = alternate.predict_log_probs(ids)
        neural_teacher = torch.logaddexp(
            primary_logp + math.log(.5375), alternate_logp + math.log(.425))
        teacher_logp = torch.logaddexp(
            neural_teacher, count_logp + math.log(.0375))
        teacher_probability = teacher_logp.exp()
    student.zero_grad(set_to_none=True)
    with training_autocast(device, precision):
        student_logp = distilled_log_probs(student, ids, log_prior)
        final_logp = torch.logaddexp(
            student_logp + math.log(.9375), count_logp + math.log(.0625))
        # BF16 expert arithmetic can leave a small row-sum drift. Keep the
        # training CE/NLL a proper distribution without altering CPU inference.
        final_logp = final_logp - final_logp.logsumexp(-1, keepdim=True)
        teacher_ce = -(teacher_probability * final_logp).sum(-1).mean()
        hard_nll = F.nll_loss(final_logp.reshape(-1, final_logp.shape[-1]),
                              labels.reshape(-1))
        loss = .75 * teacher_ce + .25 * hard_nll
    loss.backward()
    torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - started
    grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
    result = dict(protocol=PROTOCOL, purpose="stage125_train_prefix_feasibility_only",
                  source_sha256=sha(Path(__file__)), seed=125017, batch_size=24,
                  train_unigram_tokens=train_tokens,
                  ancestry={str(path): digest for path, digest in expected},
                  teacher_cross_entropy=float(teacher_ce.detach()),
                  hard_mixture_nll=float(hard_nll.detach()),
                  loss=float(loss.detach()), gradient_norm_before_clip=grad_norm,
                  teacher_max_log_normalization_error=float(
                      teacher_logp.logsumexp(-1).abs().max()),
                  student_max_log_normalization_error=float(
                      final_logp.logsumexp(-1).abs().max()),
                  count_seconds=count_seconds, gpu_forward_backward_seconds=gpu_seconds,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  finite_loss_and_gradients=(math.isfinite(float(loss.detach()))
                                             and math.isfinite(grad_norm)),
                  no_optimizer_step=True, training_prefixes_only=True,
                  no_validation_or_test_labels_in_gradients=True,
                  no_test_scoring=True)
    result["passes_feasibility_gate"] = all((
        result["finite_loss_and_gradients"],
        result["teacher_max_log_normalization_error"] < 1e-3,
        result["student_max_log_normalization_error"] < 1e-3,
        result["peak_gpu_allocated_bytes"] < 7 * 1024 ** 3,
    ))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
