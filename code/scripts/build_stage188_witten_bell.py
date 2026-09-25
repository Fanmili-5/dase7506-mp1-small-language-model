"""Train-only same-support Witten–Bell swap for the frozen Stage143 count tables."""
from __future__ import annotations

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

from common import PROTOCOL, sha
from scripts.build_kneser_ney import effective_counts
from student_ngram import NgramLM
from train_experiment import atomic_json_dump, atomic_torch_save


SOURCE = ROOT / "checkpoints/stage143-openvino-order6.pt"
SOURCE_SHA256 = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
FINAL = ROOT / "results/stage143-evidence/final.json"
VOCAB = 2048


def raw_order6_counts(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted unique (five-token history, successor, frequency)."""
    size = len(ids) - 5
    contexts = np.zeros(size, dtype=np.int64)
    for offset in range(5):
        contexts = contexts * VOCAB + ids[offset:offset + size]
    values = ids[5:5 + size]
    order = np.lexsort((values, contexts))
    contexts, values = contexts[order], values[order]
    boundary = np.r_[True, (contexts[1:] != contexts[:-1])
                     | (values[1:] != values[:-1])]
    starts = np.flatnonzero(boundary)
    return contexts[starts], values[starts], np.diff(np.r_[starts, size])


def witten_bell_same_support(
        contexts: np.ndarray, values: np.ndarray, counts: np.ndarray,
        expected: dict[str, torch.Tensor], min_count: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Recompute only probabilities, requiring exact existing CSR support."""
    if not (len(contexts) == len(values) == len(counts)):
        raise ValueError("Inconsistent count arrays")
    retained = counts >= min_count
    row_contexts = contexts[retained]
    row_values = values[retained]
    row_counts = counts[retained].astype(np.float64)
    keys, first, per_context = np.unique(
        row_contexts, return_index=True, return_counts=True)
    offsets = np.r_[0, per_context.cumsum()]
    for field, actual in (("keys", keys), ("offsets", offsets),
                          ("values", row_values)):
        if not np.array_equal(actual, expected[field].numpy()):
            raise ValueError(f"Stage143 CSR support changed: {field}")
    totals = np.add.reduceat(row_counts, first)
    denominator = totals + per_context
    mass = row_counts / np.repeat(denominator, per_context)
    backoff = per_context / denominator
    norm_error = float(np.max(np.abs(np.add.reduceat(mass, first) + backoff - 1)))
    if (norm_error > 1e-12 or np.any(mass <= 0) or np.any(backoff <= 0)
            or np.any(backoff >= 1)):
        raise ValueError("Witten–Bell probabilities do not normalize")
    masses = torch.from_numpy(mass).to(expected["mass"].dtype)
    backoffs = torch.from_numpy(backoff).to(expected["backoff"].dtype)
    rounded = (np.add.reduceat(masses.double().numpy(), first)
               + backoffs.double().numpy())
    rounded_error = float(np.max(np.abs(rounded - 1)))
    if rounded_error > 2e-6:
        raise ValueError("Float32 table roundoff exceeds normalization tolerance")
    summary = dict(contexts=len(keys), retained_edges=len(row_values),
                   max_context_normalization_error_float64=norm_error,
                   max_context_normalization_error_float32=rounded_error)
    return masses, backoffs, summary


def build(output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError("Use a new Stage188 output directory")
    if sha(SOURCE) != SOURCE_SHA256:
        raise ValueError("Protected Stage143 checkpoint changed")
    final = json.loads(FINAL.read_text(encoding="utf-8-sig"))
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("wikitext_train.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError(f"Changed fixed training file: {name}")
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if (source.get("protocol") != PROTOCOL
            or source.get("implementation") != "student_stage143_openvino_singlepass"
            or source["config"].get("max_order") != 6
            or source["config"].get("min_count") != 2):
        raise ValueError("Unexpected Stage143 count configuration")
    raw = (ROOT / "data/wikitext_train.txt").read_text(encoding="utf-8")
    ids = np.asarray(Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
                     .encode(raw).ids, dtype=np.int64)
    if len(ids) < 6 or np.any(ids < 0) or np.any(ids >= VOCAB):
        raise ValueError("Invalid train-only token sequence")
    candidate = dict(source)
    config = dict(source["config"])
    config["estimator"] = "pruned_interpolated_witten_bell_same_support"
    config.pop("discounts", None)
    candidate["config"] = config
    state = dict(source["model"])
    summary = []
    started = time.perf_counter()
    for index, order in enumerate(range(2, 7)):
        if order < 6:
            grams, counts = effective_counts(ids, order, 5)
            contexts, values = grams // VOCAB, grams % VOCAB
            count_kind = "raw" if order == 5 else "continuation"
        else:
            contexts, values, counts = raw_order6_counts(ids)
            count_kind = "raw_extension"
        prefix = f"ngram.tables.{index}."
        expected = {field: state[prefix + field] for field in (
            "keys", "offsets", "values", "mass", "backoff")}
        mass, backoff, row = witten_bell_same_support(
            contexts, values, counts, expected)
        state[prefix + "mass"] = mass
        state[prefix + "backoff"] = backoff
        summary.append(dict(order=order, count_kind=count_kind, **row))
    candidate["model"] = state
    candidate["statistics_provenance"] = dict(
        source.get("statistics_provenance", {}),
        stage188_same_support_witten_bell=dict(
            train_sha256=sha(ROOT / "data/wikitext_train.txt"),
            tokenizer_sha256=sha(ROOT / "data/tokenizer.json"),
            builder_sha256=sha(Path(__file__)), training_token_count=len(ids),
            table_summary=summary,
            note="Train-only mass/backoff replacement; neural, gate and CSR support unchanged.",
        ))

    # Count-only smoke avoids compiling the OpenVINO graph on macOS.
    count_model = NgramLM(config)
    count_state = {key.removeprefix("ngram."): value for key, value in state.items()
                   if key.startswith("ngram.")}
    count_model.load_state_dict(count_state, strict=True)
    sample = torch.from_numpy(ids[:16].copy()).reshape(2, 8)
    with torch.inference_mode():
        probabilities = count_model.distribution(sample)
        finite = bool(torch.isfinite(probabilities).all())
        normalization_error = float((probabilities.sum(-1) - 1).abs().max())
        changed_future = sample.clone()
        changed_future[:, 5:] = (changed_future[:, 5:] + 1) % VOCAB
        causal_error = float((probabilities[:, :5]
                              - count_model.distribution(changed_future)[:, :5]).abs().max())
    if not finite or normalization_error > 1e-5 or causal_error > 1e-7:
        raise ValueError("Count-only normalized causal smoke failed")

    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "stage188-witten-bell.pt"
    atomic_torch_save(candidate, checkpoint)
    asset_bytes = (final["conservative_asset_bytes"] - final["checkpoint_bytes"]
                   + checkpoint.stat().st_size)
    if asset_bytes > 64 * 1024**2:
        raise ValueError("Projected inference assets exceed course budget")
    result = dict(
        protocol=PROTOCOL, purpose="stage188_train_only_count_swap",
        status="built_not_validation_scored", source_checkpoint_sha256=SOURCE_SHA256,
        checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        builder_sha256=sha(Path(__file__)), train_sha256=sha(ROOT / "data/wikitext_train.txt"),
        tokenizer_sha256=sha(ROOT / "data/tokenizer.json"), train_tokens=len(ids),
        count_tables=summary, count_only_smoke=dict(
            finite=finite, max_normalization_error=normalization_error,
            max_causal_prefix_error=causal_error),
        projected_conservative_inference_asset_bytes=asset_bytes,
        build_seconds=time.perf_counter() - started,
        no_validation_labels_used=True, no_test_scoring=True,
        next_gate="Complete Windows CPU FP32 validation gain >=0.015 BPB versus Stage143",
    )
    atomic_json_dump(result, output_dir / "build.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
