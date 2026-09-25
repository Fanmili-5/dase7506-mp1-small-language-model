"""One synthetic train update tests physical batch memory without scoring data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from common import make_model, setup, sha
from train_experiment import training_autocast


CONFIG = ROOT / "configs/stage155_neural_budget_rdrop.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-batch", type=int, choices=(32, 16), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["width"] != 320 or config["depth"] != 10:
        raise ValueError("Unexpected Stage155 config")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    device, precision = setup("cuda", "bf16", 4)
    model, implementation_sha = make_model("student_hybrid_conv_rdrop", config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(.9, .999), weight_decay=.1)
    model.train()
    start = time.perf_counter()
    result = dict(
        purpose="synthetic_one_update_training_memory_only",
        physical_batch=args.physical_batch, effective_batch=32,
        accumulation_steps=32 // args.physical_batch,
        config_sha256=sha(CONFIG), source_sha256=sha(Path(__file__)),
        implementation_sha256=implementation_sha,
        precision=precision, device_name=torch.cuda.get_device_name(device),
        no_validation_or_test_scoring=True, no_persistent_checkpoint=True,
    )
    try:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(32 // args.physical_batch):
            ids = torch.randint(0, config["vocab"],
                                (args.physical_batch, 256), device=device)
            targets = torch.randint(0, config["vocab"], ids.shape, device=device)
            future = torch.randint(0, config["vocab"],
                                   (2, *ids.shape), device=device)
            with training_autocast(device, precision):
                loss, _ = model.rdrop_training_loss(ids, targets, future)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite synthetic loss")
            (loss / (32 // args.physical_batch)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        result["status"] = "completed_one_update"
        result["loss"] = float(loss.detach())
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "cuda_out_of_memory"
        result["error_type"] = type(exc).__name__
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
