"""Train-only compact kNN-LM quality diagnostic, validation only."""
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
from tokenizers import Tokenizer

from common import PROTOCOL, device_metrics, make_model, setup, sha, windows
from train_experiment import atomic_json_dump

BASE_SHA = "7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2"
STAGE143_SHA = "256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3"
STAGE143_CACHE_SHA = "75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3"
BASE_BPB = 1.399686162042141
N_KEYS = 100000
PROJECTION = 64
NEIGHBORS = 32
QUERY_BATCH = 512
TEMPERATURES = (.05, .10, .20)
MIXTURE_WEIGHTS = (0., .05, .10, .20, .30)


def verified_split(name: str, manifest: dict, tokenizer: Tokenizer) -> tuple[torch.Tensor, int]:
    if name not in ("train", "validation"):
        raise ValueError("Stage163 may read train and validation only")
    filename = f"wikitext_{name}.txt"
    path = ROOT / "data" / filename
    if sha(path) != manifest["sha256"][filename]:
        raise ValueError(f"Changed supplied {name} text")
    raw = path.read_bytes()
    ids = tokenizer.encode(raw.decode("utf-8")).ids
    return torch.tensor(ids, dtype=torch.long), len(raw)


def extract_hidden(model: torch.nn.Module, tokens: torch.Tensor,
                   selected: np.ndarray | None,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    count = len(selected) if selected is not None else len(tokens) - 1
    hidden = np.empty((count, 288), dtype=np.float16)
    values = np.empty(count, dtype=np.uint16)
    cursor = offset = 0
    with torch.inference_mode():
        for x, y in windows(tokens, batch_size=32):
            y_valid = y[y != -100]
            size = len(y_valid)
            if selected is not None:
                begin = int(np.searchsorted(selected, offset, side="left"))
                end = int(np.searchsorted(selected, offset + size, side="left"))
                if begin == end:
                    offset += size
                    continue
                take = torch.from_numpy(selected[begin:end] - offset).long()
            else:
                take = None
            features = model.neural.features(x.to(device)).float()
            flat = features[y.to(device) != -100]
            if take is not None:
                flat = flat[take.to(device)]
                chosen_values = y_valid[take]
            else:
                chosen_values = y_valid
            rows = len(chosen_values)
            hidden[cursor:cursor + rows] = flat.half().cpu().numpy()
            values[cursor:cursor + rows] = chosen_values.numpy().astype(np.uint16)
            cursor += rows
            offset += size
    if cursor != count or offset != len(tokens) - 1 or not np.isfinite(hidden).all():
        raise ValueError("Incomplete or non-finite causal feature extraction")
    return hidden, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--stage143-cache", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new Stage163 pilot directory")
    cache_meta = json.loads(args.stage143_cache.with_suffix(".json").read_text())
    if (sha(args.base) != BASE_SHA
            or sha(args.stage143_cache) != STAGE143_CACHE_SHA
            or cache_meta.get("array_sha256") != STAGE143_CACHE_SHA
            or cache_meta.get("checkpoint_sha256") != STAGE143_SHA
            or cache_meta.get("split") != "validation"):
        raise ValueError("Frozen Stage105/143 inputs changed")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    tokenizer_path = ROOT / "data/tokenizer.json"
    if sha(tokenizer_path) != manifest["sha256"]["tokenizer.json"]:
        raise ValueError("Supplied tokenizer changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train_tokens, train_bytes = verified_split("train", manifest, tokenizer)
    validation_tokens, validation_bytes = verified_split("validation", manifest, tokenizer)
    if len(train_tokens) != 3613343 or len(validation_tokens) - 1 != 376599 or validation_bytes != 1148007:
        raise ValueError("Unexpected fixed train/validation sizes")
    old = np.load(args.stage143_cache).astype(np.float64)
    scale = math.log(2) * validation_bytes
    if old.shape != (376599,) or abs(-float(old.sum()) / scale - BASE_BPB) > 2e-5:
        raise ValueError("Stage143 cached target stream mismatch")
    device, _ = setup("cuda", "fp32", 4)
    payload = torch.load(args.base, map_location="cpu", weights_only=True)
    if (payload.get("protocol") != PROTOCOL
            or payload.get("implementation") != "student_stage105_gated_singlepass"):
        raise ValueError("Unexpected Stage105 frozen model")
    model, module_sha = make_model(payload["implementation"], payload["config"], device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    selected = np.linspace(0, len(train_tokens) - 2, N_KEYS, dtype=np.int64)
    if len(np.unique(selected)) != N_KEYS:
        raise ValueError("Training key selection not unique")
    started = time.perf_counter()
    train_hidden, train_values = extract_hidden(model, train_tokens, selected, device)
    train_extract_seconds = time.perf_counter() - started
    started = time.perf_counter()
    query_hidden, validation_values = extract_hidden(model, validation_tokens, None, device)
    validation_extract_seconds = time.perf_counter() - started
    if not np.array_equal(validation_values, validation_tokens[1:].numpy().astype(np.uint16)):
        raise ValueError("Validation query/target alignment changed")
    del model
    torch.cuda.empty_cache()

    started = time.perf_counter()
    keys = torch.from_numpy(train_hidden).to(device=device, dtype=torch.float32)
    mean = keys.mean(0)
    centered = keys - mean
    covariance = centered.T @ centered / (N_KEYS - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -PROJECTION:].flip(1).contiguous()
    leading_eigenvalues = eigenvalues[-PROJECTION:].flip(0)
    projected_keys = F.normalize(centered @ basis, dim=-1)
    quantized_keys = (projected_keys * 127).round().clamp(-127, 127).to(torch.int8)
    quantized_key_cpu = quantized_keys.cpu().numpy()
    mean_cpu = mean.cpu().numpy()
    basis_cpu = basis.cpu().numpy()
    quantization_seconds = time.perf_counter() - started
    key_matrix = quantized_keys.to(torch.float16) / 127
    value_tensor = torch.from_numpy(train_values.astype(np.int64)).to(device)

    probabilities = np.empty((len(validation_values), len(TEMPERATURES)), dtype=np.float32)
    hits = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(validation_values), QUERY_BATCH):
            stop = min(offset + QUERY_BATCH, len(validation_values))
            query = torch.from_numpy(query_hidden[offset:stop]).to(device=device, dtype=torch.float32)
            query = F.normalize((query - mean) @ basis, dim=-1).half()
            similarity = query @ key_matrix.T
            scores, neighbor_indices = similarity.topk(NEIGHBORS, dim=-1)
            neighbors = value_tensor[neighbor_indices]
            target = torch.from_numpy(validation_values[offset:stop].astype(np.int64)).to(device)
            matching = neighbors == target[:, None]
            hits += int(matching.any(1).sum())
            for index, temperature in enumerate(TEMPERATURES):
                weight = (scores.float() / temperature).softmax(-1)
                probabilities[offset:stop, index] = (weight * matching).sum(-1).cpu().numpy()
    retrieval_seconds = time.perf_counter() - started
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1.00001).any():
        raise ValueError("Invalid normalized retrieval target probabilities")

    base_probability = np.exp(old)
    grid = []
    for temperature_index, temperature in enumerate(TEMPERATURES):
        retrieved = probabilities[:, temperature_index].astype(np.float64)
        for weight in MIXTURE_WEIGHTS:
            p = (1 - weight) * base_probability + weight * retrieved
            bpb = -float(np.log(p).sum()) / scale
            grid.append(dict(temperature=temperature, mixture_weight=weight,
                             complete_validation_bpb=bpb,
                             gain_vs_stage143_bpb=BASE_BPB - bpb))
    best = min(grid, key=lambda row: row["complete_validation_bpb"])
    args.run_dir.mkdir(parents=True)
    np.save(args.run_dir / "quantized-keys.npy", quantized_key_cpu)
    np.save(args.run_dir / "values.npy", train_values)
    np.save(args.run_dir / "projection-mean.npy", mean_cpu)
    np.save(args.run_dir / "projection-basis.npy", basis_cpu)
    np.save(args.run_dir / "selected-train-positions.npy", selected)
    asset_files = ["quantized-keys.npy", "values.npy", "projection-mean.npy", "projection-basis.npy"]
    assets = {name: dict(bytes=(args.run_dir / name).stat().st_size,
                         sha256=sha(args.run_dir / name)) for name in asset_files}
    result = dict(
        protocol=PROTOCOL, purpose="train_only_compact_knn_quality_diagnostic",
        split="validation", no_test_scoring=True,
        source_sha256=sha(Path(__file__)), stage105_base_sha256=BASE_SHA,
        stage105_implementation_sha256=module_sha,
        stage143_cache_sha256=STAGE143_CACHE_SHA,
        train_sha256=manifest["sha256"]["wikitext_train.txt"],
        validation_sha256=manifest["sha256"]["wikitext_validation.txt"],
        tokenizer_sha256=manifest["sha256"]["tokenizer.json"],
        train_tokens=len(train_tokens), train_utf8_bytes=train_bytes,
        validation_targets=len(validation_values), validation_utf8_bytes=validation_bytes,
        baseline_bpb=-float(old.sum()) / scale,
        selected_train_positions_sha256=sha(args.run_dir / "selected-train-positions.npy"),
        n_keys=N_KEYS, projection_dimensions=PROJECTION,
        neighbors=NEIGHBORS, query_batch=QUERY_BATCH,
        retrieval_hit_targets=hits, retrieval_hit_fraction=hits / len(validation_values),
        temperature_grid=list(TEMPERATURES), mixture_weight_grid=list(MIXTURE_WEIGHTS),
        full_grid=grid, best_cell=best,
        required_gain_bpb=.020,
        quality_gate_passed=best["gain_vs_stage143_bpb"] >= .020,
        top_pca_eigenvalue=float(leading_eigenvalues[0]),
        bottom_retained_pca_eigenvalue=float(leading_eigenvalues[-1]),
        train_extraction_seconds=train_extract_seconds,
        validation_extraction_seconds=validation_extract_seconds,
        quantization_seconds=quantization_seconds,
        retrieval_seconds=retrieval_seconds,
        pilot_asset_files=assets,
        pilot_asset_bytes=sum(item["bytes"] for item in assets.values()),
        conservative_stage143_plus_pilot_assets_bytes=55810412 + sum(item["bytes"] for item in assets.values()),
        warning="GPU exact-search validation diagnostic only; no CPU ANN index, full-distribution predictor or resource qualification",
        **device_metrics(device))
    atomic_json_dump(result, args.run_dir / "diagnostic.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
