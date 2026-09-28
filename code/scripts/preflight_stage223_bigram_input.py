"""Train-only data, synthetic CPU/GPU feasibility screen for Stage223."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov
import torch
from tokenizers import Tokenizer

from common import PROTOCOL, sha
from scripts.probe_stage145_attention_allocation import FeatureWrapper, make_portable, ov_model
from student_stage223_bigram_input import build_train_bigram_rank_map
from train_experiment import training_autocast
import student_stage223_bigram_rdrop as training
import student_stage223_bigram_structured as inference


REFERENCE = ROOT / "inference_assets/stage143-stage92-features.onnx"
REFERENCE_SHA = "5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4"
CONFIG = ROOT / "configs/stage223_bigram_input_rdrop.json"
CONTROL = ROOT / "configs/stage54_hybrid_conv_rdrop.json"
FROZEN = ROOT / "results/stage143-evidence/final.json"


def train_tokens() -> torch.Tensor:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unexpected data protocol")
    for name in ("tokenizer.json", "wikitext_train.txt"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Changed training data or tokenizer")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))
    raw = (ROOT / "data/wikitext_train.txt").read_bytes()
    return torch.tensor(tokenizer.encode(raw.decode("utf-8")).ids, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Choose a fresh Stage223 preflight directory")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    changed = {key for key in config.keys() | control.keys()
               if config.get(key) != control.get(key)}
    if changed != {"bigram_top_k", "bigram_dim"}:
        raise ValueError("Stage223 changed outside fixed bigram input")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Frozen reference graph changed")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8-sig"))
    if frozen["graph_sha256"] != REFERENCE_SHA or not frozen["qualified"]:
        raise ValueError("Frozen Stage143 resource evidence changed")
    tokens = train_tokens()
    mapping = build_train_bigram_rank_map(tokens)
    map_sha = __import__("hashlib").sha256(mapping.numpy().tobytes()).hexdigest()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = inference.build_model(training.inference_config(config)).eval()
    model.set_bigram_rank_map(mapping)
    ids = tokens[:8192].reshape(32, 256)
    with torch.inference_mode():
        logp = model.predict_log_probs(ids[:2, :64])
        norm_error = float(logp.logsumexp(-1).abs().max())
        changed_ids = ids[:2, :64].clone()
        changed_ids[:, 32:] = (changed_ids[:, 32:] + 97) % 2048
        causal_error = float((logp[:, :32]
                              - model.predict_log_probs(changed_ids)[:, :32]).abs().max())
        row_error = float((logp[:1]
                           - model.predict_log_probs(ids[:1, :64])).abs().max())
        rank_coverage = float((model.pair_ranks(ids) > 0).float().mean())
    eager = FeatureWrapper(model).eval()
    portable = FeatureWrapper(copy.deepcopy(model)).eval()
    rewritten = make_portable(portable)
    with torch.inference_mode():
        portable_error = float((portable(ids[:1]) - eager(ids[:1])).abs().max())
    args.run_dir.mkdir(parents=True)
    graph = args.run_dir / "stage223-train-derived-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(portable, ids, str(graph), export_params=True,
                          opset_version=17, do_constant_folding=True,
                          input_names=["ids"], output_names=["hidden"],
                          dynamic_axes={"ids": {0: "batch"}, "hidden": {0: "batch"}})
    core = ov.Core()
    reference = ov_model(core, REFERENCE)
    candidate = ov_model(core, graph)
    ov_errors = []
    with torch.inference_mode():
        for batch in (ids, ids[:1]):
            actual = torch.from_numpy(next(iter(candidate({"ids": batch.numpy()}).values())))
            ov_errors.append(float((actual - eager(batch)).abs().max()))
    times = {"reference": [], "candidate": []}
    for _ in range(2):
        reference({"ids": ids.numpy()}); candidate({"ids": ids.numpy()})
    for index in range(8):
        order = (("reference", reference), ("candidate", candidate))
        if index % 2:
            order = tuple(reversed(order))
        for label, compiled in order:
            started = time.perf_counter()
            compiled({"ids": ids.numpy()})
            times[label].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in times.items()}
    projected_seconds = (frozen["candidate_median_seconds"]
                         + 46 * (medians["candidate"] - medians["reference"]))
    projected_ratio = projected_seconds / frozen["baseline_median_seconds"]
    projected_assets = graph.stat().st_size + 25_200_000
    del model, eager, portable

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage223 requires BF16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    gpu = training.build_model(config).to(device).train()
    gpu.set_bigram_rank_map(mapping)
    optimizer = torch.optim.AdamW(gpu.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    targets = torch.randint(0, 2048, (32, 256), device=device)
    future = torch.randint(0, 2048, (2, 32, 256), device=device)
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    losses, grad_norms = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, "bf16"):
            loss, _ = gpu.rdrop_training_loss(ids.to(device), targets, future)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Stage223 loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(gpu.parameters(), 1.0))
        if not 0 < norm < float("inf"):
            raise FloatingPointError("Nonfinite Stage223 gradient")
        optimizer.step()
        losses.append(float(loss.detach())); grad_norms.append(norm)
    torch.cuda.synchronize(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    passed = bool(norm_error <= 1e-5 and causal_error <= 3e-5
                  and row_error <= 1e-5 and rewritten > 0
                  and portable_error <= 3e-4 and max(ov_errors) <= 3e-4
                  and projected_ratio <= 4.5
                  and projected_assets <= 64 * 1024**2
                  and reserved < total and reserved < 8 * 1024**3)
    result = dict(
        status="train_input_only_preflight", protocol=PROTOCOL,
        no_validation_or_test_scoring=True, config_sha256=sha(CONFIG),
        control_config_sha256=sha(CONTROL), source_sha256=sha(Path(__file__)),
        core_sha256=sha(ROOT / "student_stage223_bigram_input.py"),
        training_model_sha256=sha(ROOT / "student_stage223_bigram_rdrop.py"),
        inference_model_sha256=sha(ROOT / "student_stage223_bigram_structured.py"),
        train_bigram_rank_map_sha256=map_sha, selected_bigram_count=16384,
        rank_coverage_first_8192_training_ids=rank_coverage,
        candidate_graph_sha256=sha(graph), candidate_graph_bytes=graph.stat().st_size,
        projected_conservative_assets_bytes=projected_assets,
        max_normalization_error=norm_error, max_future_prefix_error=causal_error,
        max_independent_row_error=row_error, portable_norm_count=rewritten,
        portable_hidden_max_error=portable_error,
        openvino_hidden_max_error=max(ov_errors), timing_seconds=times,
        median_seconds=medians, projected_total_cpu_seconds=projected_seconds,
        projected_total_cpu_ratio=projected_ratio,
        synthetic_losses=losses, synthetic_grad_norms=grad_norms,
        synthetic_optimizer_steps=2,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        gpu_peak_reserved_bytes=reserved, gpu_total_bytes=total,
        admit_matched_training_pilot=passed,
    )
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Stage223 fixed feasibility gate failed")


if __name__ == "__main__":
    main()
