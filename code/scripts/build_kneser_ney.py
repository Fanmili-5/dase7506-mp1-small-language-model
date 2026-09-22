"""Build train-only pruned interpolated modified Kneser-Ney tables.

Lower orders use distinct-left-context continuation counts rather than raw
frequency. The highest order uses ordinary counts. All tables retain the same
causal CSR inference contract as the earlier absolute-discount model.
"""
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
from common import sha
from student_ngram import NgramLM
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload

VOCAB = 2048


def packed_ngrams(ids, order):
    size = len(ids) - order + 1
    if size <= 0:
        return np.zeros(0, dtype=np.int64)
    packed = np.zeros(size, dtype=np.int64)
    for offset in range(order):
        packed = packed * VOCAB + ids[offset:offset + size]
    return packed


def effective_counts(ids, order, max_order):
    """Return encoded order-grams and raw/continuation counts."""
    if order == max_order:
        return np.unique(packed_ngrams(ids, order), return_counts=True)
    # Unique (left-token, n-gram) types. Counting their n-gram suffixes gives
    # N1+(left, n-gram), the lower-order modified-KN pseudo-count.
    extended = np.unique(packed_ngrams(ids, order + 1))
    suffixes = extended % (VOCAB ** order)
    return np.unique(suffixes, return_counts=True)


def modified_discounts(counts):
    frequencies = np.bincount(np.asarray(counts, dtype=np.int64), minlength=5)
    n1, n2, n3, n4 = (float(frequencies[index]) for index in range(1, 5))
    if n1 == 0 or n2 == 0:
        return (.75, .75, .75), dict(n1=int(n1), n2=int(n2), n3=int(n3), n4=int(n4), fallback=True)
    y = n1 / (n1 + 2 * n2)
    d1 = 1 - 2 * y * n2 / n1
    d2 = 2 - 3 * y * n3 / n2 if n3 else .75
    d3 = 3 - 4 * y * n4 / n3 if n3 and n4 else .75
    discounts = (float(np.clip(d1, .01, .99)),
                 float(np.clip(d2, .01, 1.99)),
                 float(np.clip(d3, .01, 2.99)))
    return discounts, dict(n1=int(n1), n2=int(n2), n3=int(n3), n4=int(n4), y=y, fallback=False)


def fit_kneser_ney(ids, max_order=5, min_count=2):
    if not 2 <= max_order <= 5 or min_count < 1:
        raise ValueError("Invalid table parameters")
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < max_order or np.any(ids < 0) or np.any(ids >= VOCAB):
        raise ValueError("Invalid training token IDs")
    bigram_types = np.unique(packed_ngrams(ids, 2))
    continuation = np.bincount(bigram_types % VOCAB, minlength=VOCAB).astype(np.float64)
    unigram = continuation + .1
    unigram /= unigram.sum()
    tables = []
    summary = []
    for order in range(2, max_order + 1):
        grams, counts = effective_counts(ids, order, max_order)
        discounts, counts_of_counts = modified_discounts(counts)
        all_contexts, starts = np.unique(grams // VOCAB, return_index=True)
        totals = np.add.reduceat(counts, starts).astype(np.float64)
        keep = counts >= min_count
        retained = grams[keep]
        retained_counts = counts[keep]
        contexts, first, edges_per_context = np.unique(
            retained // VOCAB, return_index=True, return_counts=True)
        if len(contexts):
            context_index = np.searchsorted(all_contexts, retained // VOCAB)
            total_per_edge = totals[context_index]
            selected_discount = np.where(
                retained_counts == 1, discounts[0],
                np.where(retained_counts == 2, discounts[1], discounts[2]))
            mass = (retained_counts.astype(np.float64) - selected_discount) / total_per_edge
            backoff = 1 - np.add.reduceat(mass, first)
            if np.any(mass <= 0) or np.any(backoff < -1e-12) or np.any(backoff > 1 + 1e-12):
                raise ValueError("Invalid modified-KN probability mass")
            backoff = np.clip(backoff, 0, 1)
        else:
            mass = np.zeros(0, dtype=np.float64)
            backoff = np.zeros(0, dtype=np.float64)
        table = dict(keys=contexts, offsets=np.r_[0, edges_per_context.cumsum()],
                     values=retained % VOCAB, mass=mass, backoff=backoff)
        tables.append(table)
        summary.append(dict(order=order, count_kind="raw" if order == max_order else "continuation",
                            contexts=len(contexts), retained_edges=len(retained),
                            distinct_unpruned=len(grams), discounts=list(discounts),
                            counts_of_counts=counts_of_counts))
    config = dict(context=256, vocab=VOCAB, kind="ngram", max_order=max_order,
                  min_count=min_count, estimator="pruned_interpolated_modified_kneser_ney",
                  unigram="left_context_continuation_additive_0.1",
                  order_shapes=[[len(table["keys"]), len(table["values"])] for table in tables],
                  discounts=[row["discounts"] for row in summary])
    model = NgramLM(config)
    model.unigram.copy_(torch.from_numpy(unigram).float())
    for module, table in zip(model.tables, tables):
        for name in ("keys", "offsets", "values", "mass", "backoff"):
            getattr(module, name).copy_(torch.from_numpy(table[name]).to(getattr(module, name).dtype))
    return model, config, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-count", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    started = time.perf_counter()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for filename in ("wikitext_train.txt", "tokenizer.json"):
        if sha(ROOT / "data" / filename) != manifest["sha256"][filename]:
            raise ValueError("Changed training data/tokenizer")
    raw = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    ids = Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw).ids
    model, config, summary = fit_kneser_ney(ids, min_count=args.min_count)
    payload = checkpoint_payload(model, "student_ngram", config, 0, len(ids) - 1)
    payload["statistics_provenance"] = dict(
        train_sha256=sha(ROOT / "data/wikitext_train.txt"),
        tokenizer_sha256=sha(ROOT / "data/tokenizer.json"),
        builder_sha256=sha(Path(__file__)),
        estimator="pruned interpolated modified Kneser-Ney",
        training_token_count=len(ids), base_targets=len(ids) - 1,
        counting_passes=4, total_ngram_events=sum(len(ids) - n + 1 for n in range(2, 6)),
        note="All counts are train-only. Lower orders use distinct-left continuation counts; no validation/test fitting.",
    )
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    atomic_torch_save(payload, checkpoint)
    result = dict(config=config, orders=summary, provenance=payload["statistics_provenance"],
                  checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=sha(checkpoint),
                  build_seconds=time.perf_counter() - started, trainable_parameters=0,
                  status="train-only statistics built; quality and resources pending")
    atomic_json_dump(result, args.output_dir / "build.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
