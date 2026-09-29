"""Build a fixed low-rank input basis using only supplied MP1 training IDs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
VOCAB = 2048
DIM = 64
DISTANCE = 5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train_ids() -> np.ndarray:
    """Never read validation or test while constructing a learned asset."""
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "7506-mp1-wt2-v2":
        raise ValueError("Unexpected course protocol")
    tokenizer_path = ROOT / "data/tokenizer.json"
    train_path = ROOT / "data/wikitext_train.txt"
    for path in (tokenizer_path, train_path):
        if sha(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Fixed training input changed: {path.name}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = np.asarray(tokenizer.encode(train_path.read_text(encoding="utf-8")).ids,
                     dtype=np.int64)
    if ids.size < 257 or ids.min() < 0 or ids.max() >= VOCAB:
        raise ValueError("Unexpected training token stream")
    return ids


def count_symmetric_contexts(ids: np.ndarray, *, vocab: int = VOCAB,
                             distance: int = DISTANCE) -> np.ndarray:
    if ids.ndim != 1 or ids.size <= distance or ids.min() < 0 or ids.max() >= vocab:
        raise ValueError("Invalid train-token sequence")
    counts = np.zeros((vocab, vocab), dtype=np.float64)
    for offset in range(1, distance + 1):
        keys = ids[:-offset] * vocab + ids[offset:]
        pairs = np.bincount(keys, minlength=vocab * vocab)
        pairs = pairs.reshape(vocab, vocab)
        counts += (pairs + pairs.T) / offset
    return counts


def basis_from_counts(counts: np.ndarray, *, dim: int = DIM) -> torch.Tensor:
    if (counts.ndim != 2 or counts.shape[0] != counts.shape[1]
            or counts.shape[0] < dim or not np.isfinite(counts).all()
            or (counts < 0).any() or counts.sum() <= 0):
        raise ValueError("Invalid train-only co-occurrence table")
    marginals = counts.sum(axis=1)
    denominator = np.outer(marginals, marginals)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = counts * counts.sum() / denominator
        ppmi = np.maximum(np.log(ratio), 0)
    ppmi[~np.isfinite(ppmi)] = 0
    matrix = torch.from_numpy(ppmi.astype(np.float32))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, old_threads))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(215017)
            left, singular, _ = torch.svd_lowrank(
                matrix, q=min(counts.shape[0], dim + 16), niter=2)
    finally:
        torch.set_num_threads(old_threads)
    basis = left[:, :dim] * singular[:dim].clamp_min(0).sqrt()[None, :]
    basis -= basis.mean(dim=0, keepdim=True)
    basis /= basis.std(dim=0, keepdim=True).clamp_min(1e-6)
    basis /= dim ** 0.5
    if basis.shape != (counts.shape[0], dim) or not torch.isfinite(basis).all():
        raise ValueError("Invalid semantic basis")
    return basis.contiguous()


def build_basis() -> tuple[torch.Tensor, dict]:
    ids = load_train_ids()
    counts = count_symmetric_contexts(ids)
    basis = basis_from_counts(counts)
    metadata = {
        "purpose": "fixed_train_only_semantic_input_basis",
        "train_tokens": int(ids.size), "vocab": VOCAB,
        "dimension": DIM, "distance": DISTANCE,
        "train_sha256": sha(ROOT / "data/wikitext_train.txt"),
        "tokenizer_sha256": sha(ROOT / "data/tokenizer.json"),
        "basis_sha256": hashlib.sha256(basis.numpy().tobytes()).hexdigest(),
        "nonzero_pairs": int(np.count_nonzero(counts)),
        "basis_bytes": int(basis.numel() * basis.element_size()),
        "no_validation_or_test_read": True,
    }
    return basis, metadata


if __name__ == "__main__":
    print(json.dumps(build_basis()[1], indent=2), flush=True)
