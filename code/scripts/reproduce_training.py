"""Print or run the final training recipe; never score test or overwrite the release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]


def commands(run_dir: Path) -> list[list[str]]:
    root = run_dir.resolve()
    plan = []

    def add(script, *args):
        plan.append([sys.executable, str(CODE / "scripts" / (script + ".py")),
                     *map(str, args)])

    def average(stage, steps, epochs=False):
        directory = root / stage
        args = []
        for step in steps:
            name = f"epoch-{step:02d}.pt" if epochs else f"step-{step:06d}.pt"
            args.extend(("--checkpoint", directory / "checkpoints" / name))
        add("average_checkpoints", *args, "--output", directory / "average.pt")

    def train(stage, script, steps, *inputs):
        add(script, "--run-dir", root / stage, *inputs)
        average(stage, steps)

    late = (2400, 2700, 3000, 3300, 3600)
    first = (6000, 6300, 6600, 6900, 7200)
    continuation = (3600, 3900, 4200, 4500, 4800)
    train("stage54", "train_stage54_hybrid_conv_rdrop", first)
    parent = "stage54"
    for stage, script, steps in (
        ("stage56", "train_stage56_hybrid_conv_continuation", continuation),
        ("stage61", "train_stage61_hybrid_conv_continuation", continuation),
        ("stage63", "train_stage63_primary_emphasis", late),
        ("stage65", "train_stage65_hybrid_conv_byte_rdrop", late),
    ):
        train(stage, script, steps, "--start", root / parent / "average.pt", "--fresh-run")
        parent = stage
    add("export_stage65_hybrid_conv_byte_rdrop", "--checkpoint",
        root / "stage65/average.pt", "--output", root / "stage65/inference.pt")
    add("fit_stage67_output_bias", "--start", root / "stage65/inference.pt",
        "--run-dir", root / "stage67", "--fresh-run")
    average("stage67", (3, 4, 5), epochs=True)
    add("build_kneser_ney", "--output-dir", root / "counts5", "--min-count", 2)
    add("build_stage73_order6", "--base", root / "counts5/checkpoint.pt",
        "--output-dir", root / "counts6", "--fresh-run")
    train("stage71", "train_stage71_mixture_aware", late,
          "--start", root / "stage67/average.pt",
          "--counts", root / "counts5/checkpoint.pt", "--fresh-run")
    train("stage74", "train_stage74_balanced_hybrid_rdrop", first)
    train("stage76", "train_stage76_balanced_hybrid_continuation", continuation,
          "--start", root / "stage74/average.pt", "--fresh-run")
    add("export_stage54_hybrid_conv_rdrop", "--checkpoint",
        root / "stage76/average.pt", "--output", root / "stage76/inference.pt")
    train("stage92", "train_stage92_heterogeneous_distillation", (900, 1200, 1500, 1800),
          "--primary", root / "stage71/average.pt",
          "--alternate", root / "stage76/inference.pt",
          "--counts", root / "counts5/checkpoint.pt", "--fresh-run")
    add("fit_stage100_train_gate", "--neural", root / "stage92/average.pt",
        "--base", root / "counts5/checkpoint.pt", "--extended", root / "counts6/checkpoint.pt",
        "--output", root / "gate.json", "--fresh-run")
    add("export_retrained", "--neural", root / "stage92/average.pt",
        "--counts", root / "counts6/checkpoint.pt", "--gate", root / "gate.json",
        "--output", root / "predictor.pt")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="Print commands only; no training.")
    args = parser.parse_args()
    plan = commands(args.run_dir)
    for command in plan:
        # JSON arrays preserve paths/spaces on both Windows and POSIX.
        print(json.dumps(command), flush=True)
    if args.plan:
        return
    if args.run_dir.exists():
        parser.error("Choose a new run directory; existing outputs are never overwritten")
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "command-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for command in plan:
        subprocess.run(command, cwd=CODE, check=True)


if __name__ == "__main__":
    main()
