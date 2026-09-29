"""Cross-fit target-vocabulary frequency-group mixtures for Stage79 experts."""
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
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows
from train_experiment import atomic_json_dump


NEURAL_SHA = "20a81b19eef6784ec0b2c1935057a84e819e0f420b7e9e1187c9b117696db1a7"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
REFERENCE_BPB = 1.4030241588936954
FINE_GROUPS = 32
GROUP_COUNTS = (4, 8, 16, 32)
MIN_WEIGHT = 1e-4
MAX_WEIGHT = .50
BASE_WEIGHT = .075


def fit_weights(delta, neural_target, count_target, target_group,
                train_mask, eval_mask):
    group_count = delta.shape[1]
    base_logit = math.log(
        ((BASE_WEIGHT - MIN_WEIGHT) / (MAX_WEIGHT - MIN_WEIGHT))
        / (1 - (BASE_WEIGHT - MIN_WEIGHT) / (MAX_WEIGHT - MIN_WEIGHT)))
    logits = torch.full((group_count,), base_logit, dtype=torch.float64,
                        requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [logits], lr=.5, max_iter=80, tolerance_grad=1e-10,
        tolerance_change=1e-13, line_search_fn="strong_wolfe")
    train_delta = delta[train_mask].double()
    train_neural = neural_target[train_mask].double()
    train_count = count_target[train_mask].double()
    train_group = target_group[train_mask]

    def current_weights():
        return MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * torch.sigmoid(logits)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        weight = current_weights()
        normalization = 1 + train_delta @ weight
        target = train_neural + weight[train_group] * (train_count - train_neural)
        loss = (-target.clamp_min(1e-30).log()
                + normalization.clamp_min(1e-30).log()).mean()
        loss = loss + 1e-4 * (logits - base_logit).square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        weight = current_weights()
        eval_delta = delta[eval_mask].double()
        normalization = 1 + eval_delta @ weight
        target = (neural_target[eval_mask].double()
                  + weight[target_group[eval_mask]]
                  * (count_target[eval_mask].double()
                     - neural_target[eval_mask].double()))
        nll = -target.clamp_min(1e-30).log() + normalization.clamp_min(1e-30).log()
    return dict(
        nll_nats=float(nll.sum()), targets=int(eval_mask.sum()),
        weights=[float(value) for value in weight],
        minimum_weight=float(weight.min()), maximum_weight=float(weight.max()),
        mean_weight=float(weight.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected frozen Stage71 neural or Stage25 counts")
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    if (screen.get("protocol") != PROTOCOL
            or screen.get("neural_sha256") != NEURAL_SHA
            or screen.get("counts_sha256") != COUNTS_SHA
            or not screen.get("scalar_calibration_closed_after_this_scan")):
        raise ValueError("Expected the closed Stage79 screen")
    selected = screen["best"]
    temperature = float(selected["temperature"])
    prior_weight = float(selected["unigram_prior_weight"])
    gate_shift = float(selected["copy_gate_shift"])

    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Unexpected expert payload")
    device, _ = setup("cuda", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"]); neural.eval()
    counts.load_state_dict(count_payload["model"]); counts.eval()

    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_train.txt", "wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed data changed")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    train_text = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    train_ids = torch.tensor(tokenizer.encode(train_text).ids)
    frequencies = torch.bincount(train_ids, minlength=2048)
    order = sorted(range(2048), key=lambda token: (int(frequencies[token]), token))
    fine_group = torch.empty(2048, dtype=torch.long)
    fine_group[torch.tensor(order)] = torch.arange(2048) * FINE_GROUPS // 2048
    log_prior = ((frequencies.double() + .1)
                 / (frequencies.double().sum() + .1 * 2048)).log().float().to(device)
    fine_group_device = fine_group.to(device)

    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    validation_ids = torch.tensor(tokenizer.encode(raw.decode("utf-8")).ids)
    delta_batches, neural_batches, count_batches = [], [], []
    group_batches, half_batches = [], []
    number_of_windows = math.ceil((len(validation_ids) - 1) / 256)
    targets = 0; started = time.perf_counter()
    with torch.inference_mode():
        for batch, (x_cpu, y_cpu) in enumerate(windows(validation_ids, 16)):
            x, y = x_cpu.to(device), y_cpu.to(device)
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            hidden = neural.features(x).float()
            vocabulary = F.softmax(
                (neural.head(hidden) + neural.output_bias) / temperature
                + prior_weight * log_prior,
                dim=-1,
            )
            copy = neural.copy_distribution(hidden, x)
            gate = neural.copy_gate(hidden) + gate_shift
            neural_probability = (vocabulary * torch.sigmoid(-gate)
                                  + copy * torch.sigmoid(gate))
            count_probability = counts.predict_log_probs(x).exp()
            neural_valid = neural_probability[valid]
            count_valid = count_probability[valid]
            rows = neural_valid.shape[0]
            index = fine_group_device.unsqueeze(0).expand(rows, -1)
            neural_group = neural_valid.new_zeros(rows, FINE_GROUPS)
            count_group = count_valid.new_zeros(rows, FINE_GROUPS)
            neural_group.scatter_add_(1, index, neural_valid)
            count_group.scatter_add_(1, index, count_valid)
            delta_batches.append((count_group - neural_group).cpu())
            neural_batches.append(
                neural_probability.gather(-1, target).squeeze(-1)[valid].cpu())
            count_batches.append(
                count_probability.gather(-1, target).squeeze(-1)[valid].cpu())
            group_batches.append(fine_group_device[y[valid]].cpu())
            window_index = torch.arange(batch * 16, batch * 16 + x.shape[0])
            second = (window_index >= number_of_windows // 2).to(device)
            half_batches.append(second[:, None].expand_as(valid)[valid].cpu())
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)

    fine_delta = torch.cat(delta_batches)
    neural_target = torch.cat(neural_batches)
    count_target = torch.cat(count_batches)
    target_fine_group = torch.cat(group_batches)
    second_half = torch.cat(half_batches).bool(); first_half = ~second_half
    fixed_normalization = 1 + fine_delta.double().sum(1) * BASE_WEIGHT
    fixed_target = (neural_target.double()
                    + BASE_WEIGHT * (count_target.double() - neural_target.double()))
    fixed_nll = (-fixed_target.clamp_min(1e-30).log()
                 + fixed_normalization.clamp_min(1e-30).log()).sum()
    reference_bpb = float(fixed_nll / math.log(2) / len(raw))
    if abs(reference_bpb - REFERENCE_BPB) > 2e-5:
        raise ValueError("Frequency diagnostic does not reproduce Stage79")

    results = {}
    for group_count in GROUP_COUNTS:
        merge = FINE_GROUPS // group_count
        delta = fine_delta.view(-1, group_count, merge).sum(-1)
        target_group = target_fine_group // merge
        folds = [
            dict(train="first_half", evaluate="second_half",
                 **fit_weights(delta, neural_target, count_target, target_group,
                               first_half, second_half)),
            dict(train="second_half", evaluate="first_half",
                 **fit_weights(delta, neural_target, count_target, target_group,
                               second_half, first_half)),
        ]
        nll = sum(fold["nll_nats"] for fold in folds)
        results[str(group_count)] = dict(
            groups=group_count, folds=folds,
            crossfit_bpb=nll / math.log(2) / len(raw),
            gain_over_fixed=reference_bpb - nll / math.log(2) / len(raw),
        )
    best = min(results.values(), key=lambda row: row["crossfit_bpb"])
    group_summary = []
    for group in range(FINE_GROUPS):
        members = fine_group == group
        values = frequencies[members]
        group_summary.append(dict(
            group=group, token_types=int(members.sum()),
            minimum_train_frequency=int(values.min()),
            maximum_train_frequency=int(values.max()),
            total_train_occurrences=int(values.sum()),
        ))
    result = dict(
        protocol=PROTOCOL, split="validation", purpose="diagnostic_only_no_export",
        method="two_contiguous_half_crossfit_frequency_group_mixture",
        neural_sha256=NEURAL_SHA, counts_sha256=COUNTS_SHA,
        stage79_screen_sha256=sha(args.screen), selected_scalar_calibration=selected,
        fixed_weight=BASE_WEIGHT, fixed_bpb=reference_bpb,
        group_counts=list(GROUP_COUNTS), fine_group_summary=group_summary,
        candidates=results, best=best, targets=targets, utf8_bytes=len(raw),
        seconds=time.perf_counter() - started, validation_fit_diagnostic_only=True,
        exported_checkpoint=False, no_test_scoring=True,
        next_gate=("Only a material cross-fit gain permits fitting the same bounded "
                   "frequency mechanism from supplied training text."),
        source_sha256=sha(Path(__file__)),
    )
    if targets != 376599 or len(raw) != 1148007:
        raise ValueError("Coverage mismatch")
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
