"""Fixed one-candidate deep-supervision run with validation-only selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from train_experiment import (atomic_json_dump, atomic_torch_save, checkpoint_payload,
                              learning_rate, training_autocast)

STEPS = 7200
BATCH = 32
TARGETS = STEPS * BATCH * 256
AVERAGE_STEPS = (6000, 6300, 6600, 6900, 7200)
SOURCE_FILES = ("student_deep_supervision.py", "student_regularized.py", "student_structured.py",
    "student.py", "scripts/train_stage22_deep_supervision.py", "train_experiment.py",
    "evaluate.py", "common.py", "configs/stage22_deep_supervision.json",
    "data/manifest.json", "data/tokenizer.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    if args.run_dir.exists():
        p.error("Choose a new run directory")
    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"; checkpoints.mkdir()
    device, precision = setup("cuda", "bf16", 4)
    config_path = ROOT / "configs/stage22_deep_supervision.json"
    config = json.loads(config_path.read_text())
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    model, implementation_sha = make_model("student_deep_supervision", config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9,.999), weight_decay=.1)
    data = load_data(); tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(17)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    plan = dict(protocol=PROTOCOL, status="training", seed=17, steps=STEPS,
        primary_targets=TARGETS, unique_next_token_targets=TARGETS,
        auxiliary_label_presentations=TARGETS * len(config["deep_supervision_layers"]),
        auxiliary_weight=config["deep_supervision_weight"],
        comparison="Same seed, samples, optimizer, schedule and unique next-token targets as Stage18 H",
        precision=precision, parameters=sum(p.numel() for p in model.parameters()),
        source_hashes=sources, started_utc=datetime.now(timezone.utc).isoformat(),
        no_test_scoring=True)
    atomic_json_dump(plan, args.run_dir / "run.json")

    history, validations = [], []
    validation_seconds = 0.0
    started = time.perf_counter()
    for step in range(STEPS):
        lr = learning_rate(step, STEPS, 1e-3, 100, .1, "baseline")
        for group in optimizer.param_groups: group["lr"] = lr
        starts = torch.randint(len(tokens)-257, (BATCH,), generator=rng).to(device)
        batch = tokens[starts[:,None] + torch.arange(257, device=device)]
        optimizer.zero_grad(set_to_none=True)
        with training_autocast(device, precision):
            loss, parts = model.training_loss(batch[:,:-1], batch[:,1:])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step+1}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        completed = step + 1
        if completed % 100 == 0:
            torch.cuda.synchronize(device)
            row = dict(step=completed, loss=float(loss.detach()),
                primary_loss=float(parts["primary"]), auxiliary_loss=float(parts["auxiliary"]),
                learning_rate=lr, grad_norm=grad_norm,
                primary_targets=completed*BATCH*256,
                train_seconds=time.perf_counter()-started-validation_seconds)
            history.append(row); print(json.dumps(row), flush=True)
        if completed % 300 == 0:
            before = time.perf_counter()
            validation = score(model, *data["validation"], device, "fp32", 32)
            validation.pop("window_nll_nats")
            validation_seconds += time.perf_counter()-before
            row = dict(step=completed, **validation)
            validations.append(row); print(json.dumps({"validation":row}), flush=True)
            if completed in AVERAGE_STEPS:
                atomic_torch_save(checkpoint_payload(model,"student_deep_supervision",config,17,
                    completed*BATCH*256), checkpoints/f"step-{completed:06d}.pt")
        if completed % 300 == 0:
            atomic_json_dump(dict(completed_steps=completed,history=history,
                validation_history=validations), args.run_dir/"progress.json")
    torch.cuda.synchronize(device)
    final = checkpoint_payload(model,"student_deep_supervision",config,17,TARGETS)
    atomic_torch_save(final,args.run_dir/"checkpoint.pt")
    for name,digest in sources.items():
        if sha(ROOT/name) != digest: raise ValueError("Source changed during training: "+name)
    metrics = dict(plan,status="completed_training_validation_only",
        completed_utc=datetime.now(timezone.utc).isoformat(), train_tokens=TARGETS,
        train_seconds=time.perf_counter()-started-validation_seconds,
        validation_seconds=validation_seconds, history=history,
        validation_history=validations, final_validation=validations[-1],
        best_validation=min(validations,key=lambda r:r["bpb"]),
        checkpoint_sha256=sha(args.run_dir/"checkpoint.pt"), **device_metrics(device))
    atomic_json_dump(metrics,args.run_dir/"metrics.json")
    print(json.dumps(metrics | {"history":[],"validation_history":[]},indent=2),flush=True)


if __name__ == "__main__":
    main()
