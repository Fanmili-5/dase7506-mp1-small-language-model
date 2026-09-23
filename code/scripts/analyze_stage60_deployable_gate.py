"""Cross-fit resource-aware confidence gates for Stage56 and fixed MKN."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer

import analyze_stage50_confidence_gate as stage50
from common import PROTOCOL, make_model, setup, sha, windows
from train_experiment import atomic_json_dump

NEURAL_SHA = "b1cb9b8f7b8d94e9b8961f446b98bd80428e73818aceb4733e7af05fbaa2146b"
COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
WEIGHTS = tuple(i / 80 for i in range(17))
FEATURE_SETS = {
    "order_only": tuple(range(14, 19)),
    "sparse_prefix": tuple(range(9, 19)),
    "neural_top2_sparse_prefix": (0, 2, *range(9, 19)),
    "full_diagnostic_ceiling": tuple(range(19)),
}
MIN_WEIGHT = 1e-4


def mixture_nll(neural_target, count_target, weight):
    return -torch.logaddexp(neural_target + torch.log1p(-weight),
                            count_target + torch.log(weight))


def fit_gate(features, names, neural_target, count_target, train_mask, eval_mask):
    train_x = features[train_mask].double()
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    train_a = neural_target[train_mask].double()
    train_b = count_target[train_mask].double()
    coefficients = torch.zeros(features.shape[1], dtype=torch.float64,
                               requires_grad=True)
    bias = torch.tensor(math.log(.075 / .925), dtype=torch.float64,
                        requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coefficients, bias], lr=.5, max_iter=60, tolerance_grad=1e-9,
        tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        weight = torch.sigmoid(train_x @ coefficients + bias)
        weight = MIN_WEIGHT + (1 - 2 * MIN_WEIGHT) * weight
        loss = mixture_nll(train_a, train_b, weight).mean()
        loss = loss + 1e-5 * coefficients.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        eval_x = (features[eval_mask].double() - mean) / std
        weight = torch.sigmoid(eval_x @ coefficients + bias)
        weight = MIN_WEIGHT + (1 - 2 * MIN_WEIGHT) * weight
        nll = mixture_nll(neural_target[eval_mask].double(),
                          count_target[eval_mask].double(), weight)
    return dict(
        nll_nats=float(nll.sum()), targets=int(eval_mask.sum()),
        mean_weight=float(weight.mean()), weight_p10=float(weight.quantile(.1)),
        weight_p50=float(weight.quantile(.5)), weight_p90=float(weight.quantile(.9)),
        coefficients={name: float(value) for name, value in zip(names, coefficients)},
        bias=float(bias),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if sha(args.neural) != NEURAL_SHA or sha(args.counts) != COUNTS_SHA:
        raise ValueError("Unexpected frozen Stage56 or Stage25 checkpoint")
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_structured"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected fixed hybrid-conv neural and MKN experts")
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model(neural_payload["implementation"],
                           neural_payload["config"], device)
    counts, _ = make_model("student_ngram", count_payload["config"], device)
    neural.load_state_dict(neural_payload["model"]); neural.eval()
    counts.load_state_dict(count_payload["model"]); counts.eval()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for name in ("wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Fixed validation inputs changed")
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(
        raw.decode("utf8")).ids)
    feature_batches, neural_batches, count_batches, half_batches = [], [], [], []
    totals = torch.zeros(len(WEIGHTS), dtype=torch.float64)
    targets = 0; started = time.perf_counter()
    number_of_windows = math.ceil((len(ids) - 1) / 256)
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(ids, 32)):
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            neural_logp = neural.predict_log_probs(x)
            count_logp = counts.predict_log_probs(x)
            all_features = stage50.confidence_features(
                neural_logp, count_logp, counts, x)[valid.flatten()]
            neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid]
            count_target = count_logp.gather(-1, target).squeeze(-1)[valid]
            feature_batches.append(all_features.cpu())
            neural_batches.append(neural_target.cpu())
            count_batches.append(count_target.cpu())
            indices = torch.arange(batch * 32, batch * 32 + x.shape[0])
            halves = (indices >= number_of_windows // 2).unsqueeze(1).expand_as(valid)
            half_batches.append(halves[valid].cpu())
            for index, scalar in enumerate(WEIGHTS):
                if scalar == 0:
                    totals[index] -= neural_target.double().sum()
                else:
                    weight = torch.full_like(neural_target, scalar).double()
                    totals[index] += mixture_nll(neural_target.double(),
                                                 count_target.double(), weight).sum()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    all_features = torch.cat(feature_batches)
    neural_target = torch.cat(neural_batches)
    count_target = torch.cat(count_batches)
    second_half = torch.cat(half_batches).bool(); first_half = ~second_half
    feature_results = {}
    for label, columns in FEATURE_SETS.items():
        names = tuple(stage50.FEATURE_NAMES[index] for index in columns)
        features = all_features[:, columns]
        folds = [
            dict(train="first_half", evaluate="second_half",
                 **fit_gate(features, names, neural_target, count_target,
                            first_half, second_half)),
            dict(train="second_half", evaluate="first_half",
                 **fit_gate(features, names, neural_target, count_target,
                            second_half, first_half)),
        ]
        nll = sum(fold["nll_nats"] for fold in folds)
        feature_results[label] = dict(
            feature_names=list(names), folds=folds,
            crossfit_bpb=nll / math.log(2) / len(raw),
            deployability=("low_overhead_candidate" if label != "full_diagnostic_ceiling"
                           else "diagnostic_ceiling_requires_dense_count_statistics"),
        )
    rows = [dict(weight=weight, nll_nats=float(total),
                 bpb=float(total / math.log(2) / len(raw)))
            for weight, total in zip(WEIGHTS, totals)]
    result = dict(
        protocol=PROTOCOL, split="validation", purpose="diagnostic_only_no_export",
        neural_sha256=NEURAL_SHA, counts_sha256=COUNTS_SHA,
        fixed_weight_grid=rows, best_fixed=min(rows, key=lambda row: row["bpb"]),
        feature_sets=feature_results, targets=targets, utf8_bytes=len(raw),
        seconds=time.perf_counter() - started, validation_fit_diagnostic_only=True,
        exported_checkpoint=False, no_test_scoring=True,
        next_gate=("Train a proxy neural and temporary MKN excluding a calibration slice "
                   "only if a low-overhead feature set materially beats fixed weight."),
        source_sha256=sha(Path(__file__)),
    )
    if targets != 376599 or len(raw) != 1148007 or all_features.shape[1] != 19:
        raise ValueError("Coverage or feature shape mismatch")
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
