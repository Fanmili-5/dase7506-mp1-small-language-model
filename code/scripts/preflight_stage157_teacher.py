"""Hash-pin and time one train-only Stage105/155-to-Stage155 update."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from train_experiment import atomic_json_dump, training_autocast

STAGE105_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"
STAGE155_SHA = "c2fe32ba15b0f13b11b43247f058971dac5717ac37fd2acda1bbaa173277bce1"
CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"


def load_checkpoint(path: Path, expected_sha: str, device: torch.device):
    if sha(path) != expected_sha:
        raise ValueError(f"Unexpected checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected course protocol")
    model, implementation_sha = make_model(payload["implementation"],
                                            payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload, implementation_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage105", type=Path, required=True)
    parser.add_argument("--stage155", type=Path, required=True)
    parser.add_argument("--stage143-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new preflight output")
    cache_meta = json.loads(args.stage143_cache.with_suffix(".json").read_text())
    if (sha(args.stage143_cache) != CACHE_SHA
            or cache_meta.get("array_sha256") != CACHE_SHA):
        raise ValueError("Unexpected Stage143 validation diagnostic cache")
    device, precision = setup("cuda", "bf16", 4)
    if device.type != "cuda":
        raise ValueError("Expected CUDA memory preflight")
    teacher_old, old_payload, old_module_sha = load_checkpoint(
        args.stage105, STAGE105_SHA, device)
    teacher_new, new_payload, new_module_sha = load_checkpoint(
        args.stage155, STAGE155_SHA, device)
    if (old_payload["implementation"] != "student_stage105_gated_singlepass"
            or new_payload["implementation"] != "student_hybrid_conv_rdrop"):
        raise ValueError("Unexpected frozen teacher architecture")
    student, _ = make_model(new_payload["implementation"],
                            new_payload["config"], device)
    student.load_state_dict(new_payload["model"], strict=True)
    for teacher in (teacher_old, teacher_new):
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    data = load_data()
    first_x, first_y = next(windows(data["validation"][0], batch_size=8))
    first_x, first_y = first_x.to(device), first_y.to(device)
    with torch.inference_mode():
        old_logp = teacher_old.predict_log_probs(first_x).float()
        old_target = old_logp.gather(-1, first_y.clamp_min(0).unsqueeze(-1))
        old_target = old_target.squeeze(-1)[first_y != -100].cpu().numpy()
    cached = np.load(args.stage143_cache)[:len(old_target)]
    target_error = float(np.max(np.abs(old_target - cached)))

    tokens = data["train"][0]
    rng = torch.Generator().manual_seed(157017)
    starts = torch.randint(len(tokens) - 257, (16,), generator=rng)
    sequence = tokens[starts[:, None] + torch.arange(257)]
    ids, targets = sequence[:, :256].to(device), sequence[:, 1:].to(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        old_logp = teacher_old.predict_log_probs(ids).float()
        new_logp = teacher_new.predict_log_probs(ids).float()
        teacher_logp = torch.logaddexp(old_logp, new_logp) - math.log(2)
        normalization_error = float(teacher_logp.logsumexp(-1).abs().max())
        teacher_probability = teacher_logp.exp()
    student.train()
    with training_autocast(device, precision):
        student_logp = student.predict_log_probs(ids).float()
        distill = -(teacher_probability * student_logp).sum(-1).mean()
        hard = F.nll_loss(student_logp.reshape(-1, 2048), targets.reshape(-1))
        loss = .75 * distill + .25 * hard
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Stage157 pilot loss")
    loss.backward()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    result = dict(
        protocol=PROTOCOL, stage105_sha256=STAGE105_SHA,
        stage155_sha256=STAGE155_SHA, stage143_cache_sha256=CACHE_SHA,
        stage105_module_sha256=old_module_sha,
        stage155_module_sha256=new_module_sha,
        source_sha256=sha(Path(__file__)),
        max_stage105_vs_stage143_target_logp_error=target_error,
        max_teacher_log_normalization_error=normalization_error,
        one_step_wall_seconds=seconds, peak_cuda_allocated_bytes=peak,
        loss=float(loss.detach()), distill_loss=float(distill.detach()),
        hard_loss=float(hard.detach()), physical_batch=16,
        source_split="train", no_test_scoring=True,
        pass_target_parity=target_error <= 3e-4,
        pass_normalization=normalization_error <= 1e-3,
        pass_memory=peak <= 6_500_000_000,
        pass_speed=seconds <= 3.0,
    )
    result["pilot_authorized"] = all(result[key] for key in (
        "pass_target_parity", "pass_normalization", "pass_memory", "pass_speed"))
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
