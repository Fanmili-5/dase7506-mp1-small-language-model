"""Read-only Stage92/129 teacher-transfer audit on train suffix and validation."""
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

from common import PROTOCOL, load_data, make_model, setup, sha, windows
from scripts.analyze_stage98_distilled_gate import NEURAL_SHA, distilled_log_probs
from scripts.fit_stage100_train_gate import BASE_SHA
from scripts.train_stage86_calibration_aware import calibrated_neural_log_probs, train_log_prior
from scripts.train_stage92_heterogeneous_distillation import PRIMARY_SHA, ALTERNATE_SHA
from student_mixture_aware import build_target_edge_keys, count_target_probability
from train_experiment import atomic_json_dump


STAGE129_SHA = "840cadf5224b9e63fadd0f9434a0a957867c52111693925575f11eddc7829f7c"
EXPECTED_TARGETS, EXPECTED_BYTES = 376599, 1148007
EXPECTED_STAGE92, EXPECTED_STAGE129 = 1.4017076627486786, 1.4019352814472137


def audit_split(name, tokens, raw_bytes, models, counts, keys, prior, device):
    started = time.perf_counter()
    totals = {key: 0.0 for key in (
        "student92_neural_nll", "student129_neural_nll",
        "student92_mixture_nll", "student129_mixture_nll",
        "teacher_neural_nll", "teacher_ce_to_student92",
        "teacher_ce_to_student129", "teacher_entropy")}
    targets = 0
    windows_improved = 0
    window_count = 0
    with torch.inference_mode():
        for batch, (ids_cpu, labels_cpu) in enumerate(windows(tokens, 32)):
            count_prob_cpu = count_target_probability(
                counts, ids_cpu, labels_cpu, keys)
            ids, labels = ids_cpu.to(device), labels_cpu.to(device)
            count_prob = count_prob_cpu.to(device)
            valid = labels != -100
            target = labels.clamp_min(0).unsqueeze(-1)
            primary = calibrated_neural_log_probs(models["primary"], ids, prior)
            alternate = models["alternate"].predict_log_probs(ids)
            teacher_logp = torch.logaddexp(
                primary + math.log(.55), alternate + math.log(.45))
            teacher_prob = teacher_logp.exp()
            teacher_prob = teacher_prob / teacher_prob.sum(-1, keepdim=True)
            student92 = distilled_log_probs(models["student92"], ids, prior)
            student129 = distilled_log_probs(models["student129"], ids, prior)
            score92 = student92.gather(-1, target).squeeze(-1)
            score129 = student129.gather(-1, target).squeeze(-1)
            score_teacher = teacher_logp.gather(-1, target).squeeze(-1)
            count_logp = count_prob.clamp_min(torch.finfo(torch.float32).tiny).log()
            mix92 = torch.logaddexp(score92 + math.log(.9375),
                                    count_logp + math.log(.0625))
            mix129 = torch.logaddexp(score129 + math.log(.9375),
                                     count_logp + math.log(.0625))
            contributions = dict(
                student92_neural_nll=-score92,
                student129_neural_nll=-score129,
                student92_mixture_nll=-mix92,
                student129_mixture_nll=-mix129,
                teacher_neural_nll=-score_teacher,
                teacher_ce_to_student92=-(teacher_prob * student92).sum(-1),
                teacher_ce_to_student129=-(teacher_prob * student129).sum(-1),
                teacher_entropy=-(teacher_prob * teacher_prob.clamp_min(1e-30).log()).sum(-1),
            )
            for metric, values in contributions.items():
                totals[metric] += float(values[valid].double().sum())
            per_window_gain = ((-mix92 + mix129) * valid).sum(-1)
            windows_improved += int((per_window_gain > 0).sum())
            window_count += ids.shape[0]
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(split=name, batch=batch, targets=targets)),
                      flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = dict(split=name, targets=targets, utf8_bytes=raw_bytes,
                  totals_nats=totals,
                  nats_per_target={key: value / targets for key, value in totals.items()},
                  teacher_kl_to_student92_nats_per_target=(
                      totals["teacher_ce_to_student92"] - totals["teacher_entropy"]) / targets,
                  teacher_kl_to_student129_nats_per_target=(
                      totals["teacher_ce_to_student129"] - totals["teacher_entropy"]) / targets,
                  window_count=window_count,
                  windows_improved_by_stage129=windows_improved,
                  seconds=time.perf_counter() - started)
    if raw_bytes is not None:
        result["bpb"] = {
            key: totals[key] / math.log(2) / raw_bytes
            for key in ("student92_mixture_nll", "student129_mixture_nll")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student92", type=Path, required=True)
    parser.add_argument("--student129", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new Stage130 output path")
    expected = dict(student92=NEURAL_SHA, student129=STAGE129_SHA,
                    primary=PRIMARY_SHA, alternate=ALTERNATE_SHA,
                    counts=BASE_SHA)
    paths = {name: getattr(args, name) for name in expected}
    if any(sha(paths[name]) != digest for name, digest in expected.items()):
        raise ValueError("Stage130 frozen checkpoint ancestry mismatch")
    payloads = {name: torch.load(path, map_location="cpu", weights_only=True)
                for name, path in paths.items()}
    if any(payload.get("protocol") != PROTOCOL for payload in payloads.values()):
        raise ValueError("Stage130 checkpoint protocol mismatch")
    device, precision = setup("cuda", "fp32", 4)
    models = {}
    for name in ("student92", "student129", "primary", "alternate"):
        payload = payloads[name]
        model, _ = make_model(payload["implementation"], payload["config"], device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        models[name] = model
    counts_payload = payloads["counts"]
    counts, _ = make_model("student_ngram", counts_payload["config"], torch.device("cpu"))
    counts.load_state_dict(counts_payload["model"], strict=True)
    counts.eval()
    keys = build_target_edge_keys(counts)
    prior, prior_tokens = train_log_prior()
    prior = prior.to(device)
    data = load_data()
    train = data["train"][0]
    cut90 = int(len(train) * .90)
    train_result = audit_split("train_last_10_percent_in_sample", train[cut90:],
                               None, models, counts, keys, prior, device)
    validation, raw_bytes = data["validation"]
    validation_result = audit_split("validation", validation, raw_bytes,
                                    models, counts, keys, prior, device)
    if (validation_result["targets"] != EXPECTED_TARGETS
            or raw_bytes != EXPECTED_BYTES
            or abs(validation_result["bpb"]["student92_mixture_nll"]
                   - EXPECTED_STAGE92) > 2e-5
            or abs(validation_result["bpb"]["student129_mixture_nll"]
                   - EXPECTED_STAGE129) > 2e-5):
        raise ValueError("Stage130 validation control failed")
    result = dict(protocol=PROTOCOL, purpose="stage130_read_only_transfer_audit",
                  precision=precision, source_sha256=sha(Path(__file__)),
                  checkpoint_sha256=expected, train_token_count=len(train),
                  train_suffix_begin=cut90, train_suffix_is_not_held_out=True,
                  train_unigram_tokens=prior_tokens,
                  train=train_result, validation=validation_result,
                  no_new_training=True, no_checkpoint_export=True,
                  no_test_scoring=True)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
