"""Validation-only cross-fit diagnostic for an input-visible neural/count gate.

The fitted gates are never serialized into a submission checkpoint.  This stage
asks whether legal confidence features can generalize across two disjoint,
contiguous validation halves before paying for train-only proxy calibration.
"""
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


FEATURE_NAMES = (
    "neural_max_logp", "count_max_logp", "neural_margin", "count_margin",
    "neural_entropy_norm", "count_entropy_norm", "top1_agreement",
    "neural_logp_at_count_top1", "count_logp_at_neural_top1",
    "position_fraction", "highest_backoff", "highest_log_edges",
    "highest_max_mass", "highest_total_mass",
    "order_1", "order_2", "order_3", "order_4", "order_5",
)
WEIGHTS = tuple(i / 40 for i in range(21))  # count weights 0 through .50
MIN_WEIGHT = 1e-4


def count_context_features(model, ids):
    """Return target-independent sparse-table confidence for every prefix."""
    batch, length = ids.shape
    positions = torch.arange(length).expand(batch, -1).flatten()
    order = torch.ones(ids.numel(), dtype=torch.long)
    backoff = torch.ones(ids.numel(), dtype=torch.float32)
    edges_out = torch.zeros(ids.numel(), dtype=torch.float32)
    max_mass = torch.zeros(ids.numel(), dtype=torch.float32)
    total_mass = torch.zeros(ids.numel(), dtype=torch.float32)
    for history, table in enumerate(model.tables, 1):
        if history > length or table.keys.numel() == 0:
            continue
        keys = torch.zeros_like(ids)
        for lag in range(history - 1, -1, -1):
            shifted = ids if lag == 0 else F.pad(ids[:, :-lag], (lag, 0))
            keys = keys * model.vocab + shifted
        keys = keys.flatten()
        locations = torch.searchsorted(table.keys, keys).clamp_max(table.keys.numel() - 1)
        found = (table.keys[locations] == keys) & (positions >= history - 1)
        if not found.any():
            continue
        found_rows = found.nonzero().flatten()
        found_locations = locations[found_rows]
        sizes = table.offsets[found_locations + 1] - table.offsets[found_locations]
        starts = table.offsets[found_locations]
        repeated_rows = torch.repeat_interleave(found_rows, sizes)
        repeated_starts = torch.repeat_interleave(starts, sizes)
        local = torch.arange(repeated_rows.numel()) - torch.repeat_interleave(
            sizes.cumsum(0) - sizes, sizes)
        mass = table.mass[repeated_starts + local]
        totals = torch.zeros(ids.numel(), dtype=torch.float32)
        totals.scatter_add_(0, repeated_rows, mass)
        maxima = torch.zeros(ids.numel(), dtype=torch.float32)
        maxima.scatter_reduce_(0, repeated_rows, mass, reduce="amax", include_self=True)
        order[found_rows] = history + 1
        backoff[found_rows] = table.backoff[found_locations]
        edges_out[found_rows] = sizes.float()
        max_mass[found_rows] = maxima[found_rows]
        total_mass[found_rows] = totals[found_rows]
    return order, backoff, edges_out, max_mass, total_mass


def confidence_features(neural_logp, count_logp, count_model, ids):
    """Build legal features from two predictive distributions and the prefix."""
    neural_top = neural_logp.topk(2, dim=-1)
    count_top = count_logp.topk(2, dim=-1)
    neural_prob = neural_logp.exp()
    count_prob = count_logp.exp()
    neural_entropy = -(neural_prob * neural_logp).sum(-1) / math.log(neural_logp.shape[-1])
    count_entropy = -(count_prob * count_logp).sum(-1) / math.log(count_logp.shape[-1])
    neural_at_count = neural_logp.gather(-1, count_top.indices[..., :1]).squeeze(-1)
    count_at_neural = count_logp.gather(-1, neural_top.indices[..., :1]).squeeze(-1)
    order, backoff, edges, max_mass, total_mass = count_context_features(count_model, ids)
    position = torch.arange(ids.shape[1]).expand_as(ids).float() / max(1, ids.shape[1] - 1)
    one_hot_order = F.one_hot(order - 1, num_classes=5).float()
    flat = lambda value: value.reshape(-1).float()
    columns = (
        flat(neural_top.values[..., 0]), flat(count_top.values[..., 0]),
        flat(neural_top.values[..., 0] - neural_top.values[..., 1]),
        flat(count_top.values[..., 0] - count_top.values[..., 1]),
        flat(neural_entropy), flat(count_entropy),
        flat(neural_top.indices[..., 0] == count_top.indices[..., 0]),
        flat(neural_at_count), flat(count_at_neural), flat(position),
        backoff, torch.log1p(edges), max_mass, total_mass,
    )
    return torch.cat((torch.stack(columns, dim=1), one_hot_order), dim=1)


def mixture_nll(neural_target, count_target, weight):
    return -torch.logaddexp(neural_target + torch.log1p(-weight),
                            count_target + torch.log(weight))


def fit_linear_gate(features, neural_target, count_target, train_mask, eval_mask):
    train_x = features[train_mask].double()
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    train_a = neural_target[train_mask].double()
    train_b = count_target[train_mask].double()
    coefficients = torch.zeros(features.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(math.log(.125 / .875), dtype=torch.float64, requires_grad=True)
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
        coefficients={name: float(value) for name, value in zip(FEATURE_NAMES, coefficients)},
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
    neural_sha, counts_sha = sha(args.neural), sha(args.counts)
    neural_payload = torch.load(args.neural, map_location="cpu", weights_only=True)
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if (neural_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_structured"
            or count_payload.get("protocol") != PROTOCOL
            or count_payload.get("implementation") != "student_ngram"):
        raise ValueError("Expected fixed-protocol structured neural and count experts")
    device, _ = setup("cpu", "fp32", 4)
    neural, _ = make_model("student_structured", neural_payload["config"], device)
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
    targets = 0
    started = time.perf_counter()
    number_of_windows = math.ceil((len(ids) - 1) / 256)
    with torch.inference_mode():
        for batch, (x, y) in enumerate(windows(ids, 32)):
            valid = y != -100
            target = y.clamp_min(0).unsqueeze(-1)
            neural_logp = neural.predict_log_probs(x)
            count_logp = counts.predict_log_probs(x)
            features = confidence_features(neural_logp, count_logp, counts, x)[valid.flatten()]
            neural_target = neural_logp.gather(-1, target).squeeze(-1)[valid]
            count_target = count_logp.gather(-1, target).squeeze(-1)[valid]
            feature_batches.append(features.cpu())
            neural_batches.append(neural_target.cpu())
            count_batches.append(count_target.cpu())
            window_indices = torch.arange(batch * 32, batch * 32 + x.shape[0])
            halves = (window_indices >= number_of_windows // 2).unsqueeze(1).expand_as(valid)
            half_batches.append(halves[valid].cpu())
            for i, weight in enumerate(WEIGHTS):
                if weight == 0:
                    totals[i] -= neural_target.double().sum()
                else:
                    w = torch.full_like(neural_target, weight)
                    totals[i] += mixture_nll(neural_target.double(), count_target.double(), w.double()).sum()
            targets += int(valid.sum())
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    features = torch.cat(feature_batches)
    neural_target = torch.cat(neural_batches)
    count_target = torch.cat(count_batches)
    second_half = torch.cat(half_batches).bool()
    first_half = ~second_half
    folds = [
        dict(train="first_half", evaluate="second_half",
             **fit_linear_gate(features, neural_target, count_target, first_half, second_half)),
        dict(train="second_half", evaluate="first_half",
             **fit_linear_gate(features, neural_target, count_target, second_half, first_half)),
    ]
    crossfit_nll = sum(fold["nll_nats"] for fold in folds)
    rows = [dict(weight=weight, nll_nats=float(total),
                 bpb=float(total / math.log(2) / len(raw)))
            for weight, total in zip(WEIGHTS, totals)]
    result = dict(
        protocol=PROTOCOL, split="validation", purpose="diagnostic_only_no_export",
        method="two_contiguous_half_crossfit_linear_confidence_gate",
        neural_sha256=neural_sha, counts_sha256=counts_sha,
        feature_names=list(FEATURE_NAMES), fixed_weight_grid=rows,
        best_fixed=min(rows, key=lambda row: row["bpb"]), folds=folds,
        crossfit_bpb=crossfit_nll / math.log(2) / len(raw),
        targets=targets, utf8_bytes=len(raw), seconds=time.perf_counter() - started,
        validation_fit_diagnostic_only=True, exported_checkpoint=False,
        next_gate=("Only if cross-fit materially beats the fixed mixture, fit the same gate "
                   "using a separately trained train-only proxy neural expert."),
        source_sha256=sha(Path(__file__)),
    )
    if targets != 376599 or len(raw) != 1148007 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Coverage or feature shape mismatch")
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
