"""Build train-only order-2..5 pruned absolute-discount tables, not Kneser-Ney.

Reads only the supplied train text and tokenizer; verifies both against manifest.
Pruned mass backs off, rather than renormalizing surviving counts to certainty.
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


def fit_tables(ids, max_order=5, min_count=3, discount=0.75):
    if not 2 <= max_order <= 5 or min_count < 1 or not 0 < discount < 1:
        raise ValueError("Invalid table parameters")
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < max_order or np.any(ids < 0) or np.any(ids >= 2048):
        raise ValueError("Invalid training token IDs")
    unigram = np.bincount(ids, minlength=2048).astype(np.float64) + 0.1
    unigram /= unigram.sum()
    tables = []
    for order in range(2, max_order + 1):
        size = len(ids) - order + 1
        packed = np.zeros(size, dtype=np.int64)
        for j in range(order):
            packed = packed * 2048 + ids[j:j + size]
        grams, counts = np.unique(packed, return_counts=True)
        all_contexts, starts = np.unique(grams // 2048, return_index=True)
        totals = np.add.reduceat(counts, starts)
        keep = counts >= min_count
        retained, retained_counts = grams[keep], counts[keep]
        contexts, first, edges_per_context = np.unique(retained // 2048, return_index=True, return_counts=True)
        if len(contexts):
            total_per_edge = totals[np.searchsorted(all_contexts, retained // 2048)]
            mass = (retained_counts.astype(np.float64) - discount) / total_per_edge
            backoff = 1 - np.add.reduceat(mass, first)
        else:
            mass, backoff = np.zeros(0), np.zeros(0)
        tables.append(dict(keys=contexts, offsets=np.r_[0, edges_per_context.cumsum()],
                           values=retained % 2048, mass=mass, backoff=backoff,
                           distinct_unpruned=int(len(grams))))
    config = dict(context=256, vocab=2048, kind="ngram", max_order=max_order,
                  min_count=min_count, discount=discount, unigram_additive=0.1,
                  order_shapes=[[len(t["keys"]), len(t["values"])] for t in tables])
    model = NgramLM(config)
    model.unigram.copy_(torch.from_numpy(unigram).float())
    for module, table in zip(model.tables, tables):
        for name in ("keys", "offsets", "values", "mass", "backoff"):
            buffer = getattr(module, name)
            buffer.copy_(torch.from_numpy(table[name]).to(buffer.dtype))
    summary = [dict(order=i + 2, contexts=len(t["keys"]), retained_edges=len(t["values"]),
                    distinct_unpruned=t["distinct_unpruned"]) for i, t in enumerate(tables)]
    return model, config, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    start = time.perf_counter()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for filename in ("wikitext_train.txt", "tokenizer.json"):
        if sha(ROOT / "data" / filename) != manifest["sha256"][filename]:
            raise ValueError("Changed training data/tokenizer")
    raw = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    ids = Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw).ids
    model, config, summary = fit_tables(ids)
    payload = checkpoint_payload(model, "student_ngram", config, 0, len(ids) - 1)
    payload["statistics_provenance"] = dict(train_sha256=sha(ROOT / "data/wikitext_train.txt"),
        tokenizer_sha256=sha(ROOT / "data/tokenizer.json"), builder_sha256=sha(Path(__file__)),
        estimator="pruned interpolated absolute discount", training_token_count=len(ids),
        base_targets=len(ids)-1, counting_passes=4,
        total_ngram_events=sum(len(ids)-n+1 for n in range(2,6)),
        note="Counts are train-only; no validation fitting. Counting work is separate from gradient targets.")
    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    atomic_torch_save(payload, checkpoint)
    result = dict(config=config, orders=summary, provenance=payload["statistics_provenance"],
                  checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=sha(checkpoint),
                  build_seconds=time.perf_counter()-start, trainable_parameters=0,
                  status="standalone statistical diagnostic, not claimed as a trainable submission")
    atomic_json_dump(result, args.output_dir / "build.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
