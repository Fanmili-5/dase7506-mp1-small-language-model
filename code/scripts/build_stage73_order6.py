"""Append a train-only raw order-6 MKN table to the frozen Stage25 estimator."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, make_model, sha
from scripts.build_kneser_ney import modified_discounts
from train_experiment import atomic_json_dump, atomic_torch_save

BASE_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
VOCAB = 2048


def order6_table(ids, min_count=2):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 6 or min_count < 1:
        raise ValueError("Invalid token sequence or pruning threshold")
    size = len(ids) - 5
    contexts = np.zeros(size, dtype=np.int64)
    for offset in range(5):
        contexts = contexts * VOCAB + ids[offset:offset + size]
    values = ids[5:5 + size]
    order = np.lexsort((values, contexts))
    sorted_contexts, sorted_values = contexts[order], values[order]
    boundary = np.r_[True, (sorted_contexts[1:] != sorted_contexts[:-1])
                     | (sorted_values[1:] != sorted_values[:-1])]
    starts = np.flatnonzero(boundary)
    pair_contexts, pair_values = sorted_contexts[starts], sorted_values[starts]
    counts = np.diff(np.r_[starts, size])
    discounts, count_evidence = modified_discounts(counts)
    all_contexts, total_starts = np.unique(pair_contexts, return_index=True)
    totals = np.add.reduceat(counts, total_starts).astype(np.float64)
    keep = counts >= min_count
    retained_contexts = pair_contexts[keep]
    retained_values = pair_values[keep]
    retained_counts = counts[keep]
    contexts_out, first, per_context = np.unique(
        retained_contexts, return_index=True, return_counts=True
    )
    total_per_edge = totals[np.searchsorted(all_contexts, retained_contexts)]
    selected_discount = np.where(
        retained_counts == 1, discounts[0],
        np.where(retained_counts == 2, discounts[1], discounts[2]),
    )
    mass = (retained_counts.astype(np.float64) - selected_discount) / total_per_edge
    backoff = 1 - np.add.reduceat(mass, first)
    if (np.any(mass <= 0) or np.any(backoff < -1e-12)
            or np.any(backoff > 1 + 1e-12)):
        raise ValueError("Invalid order-6 probability mass")
    table = dict(
        keys=contexts_out,
        offsets=np.r_[0, per_context.cumsum()],
        values=retained_values,
        mass=mass,
        backoff=np.clip(backoff, 0, 1),
    )
    summary = dict(
        order=6, count_kind="raw_extension", contexts=len(contexts_out),
        retained_edges=len(retained_values), distinct_unpruned=len(counts),
        discounts=list(discounts), counts_of_counts=count_evidence,
    )
    return table, summary


def extend_model(base_payload, ids, min_count=2):
    if (base_payload.get("protocol") != PROTOCOL
            or base_payload.get("implementation") != "student_ngram"
            or base_payload["config"].get("max_order") != 5):
        raise ValueError("Expected the frozen order-5 count estimator")
    table, summary = order6_table(ids, min_count)
    config = dict(base_payload["config"])
    config["max_order"] = 6
    config["order6_extension_min_count"] = min_count
    config["order_shapes"] = list(config["order_shapes"]) + [
        [len(table["keys"]), len(table["values"])]
    ]
    config["discounts"] = list(config["discounts"]) + [summary["discounts"]]
    model, _ = make_model("student_ngram", config, torch.device("cpu"))
    state = model.state_dict()
    for name, value in base_payload["model"].items():
        state[name] = value
    prefix = f"tables.{len(model.tables) - 1}."
    for name in ("keys", "offsets", "values", "mass", "backoff"):
        state[prefix + name] = torch.from_numpy(table[name]).to(
            state[prefix + name].dtype
        )
    model.load_state_dict(state, strict=True)
    return model, config, summary


def main():
    global BASE_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fresh-run", action="store_true",
                        help="Accept newly trained inputs; keep protocol checks and record actual hashes.")
    args = parser.parse_args()
    if args.fresh_run:
        BASE_SHA = sha(args.base)
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    if sha(args.base) != BASE_SHA:
        raise ValueError("Unexpected Stage25 count checkpoint")
    started = time.perf_counter()
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("wikitext_train.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Changed training data/tokenizer")
    raw = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    ids = Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw).ids
    model, config, summary = extend_model(base, ids, min_count=2)
    payload = dict(base, config=config, model=model.state_dict())
    payload["statistics_provenance"] = dict(
        base.get("statistics_provenance", {}),
        order6_extension=dict(
            train_sha256=sha(ROOT / "data/wikitext_train.txt"),
            tokenizer_sha256=sha(ROOT / "data/tokenizer.json"),
            builder_sha256=sha(Path(__file__)), min_count=2,
            training_token_count=len(ids), raw_events=len(ids) - 5,
            summary=summary,
            note="Train-only raw order-6 extension; lower Stage25 tables unchanged.",
        ),
    )
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    atomic_torch_save(payload, checkpoint)
    result = dict(
        protocol=PROTOCOL, config=config, order6=summary,
        base_checkpoint_sha256=BASE_SHA, checkpoint_sha256=sha(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,
        build_seconds=time.perf_counter() - started, no_test_scoring=True,
    )
    atomic_json_dump(result, args.output_dir / "build.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
