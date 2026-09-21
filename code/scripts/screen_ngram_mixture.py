"""Fixed validation-only mixture grid, then independent official-score check.

Neural predictions are reused within each batch, not saved as predictor assets.
No target is ever passed to either model. Counts must derive only from train.
This diagnostic scan's elapsed time is NOT a candidate CPU benchmark.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, make_model, setup, sha, windows
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload
from student_ngram import build_model

WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.35)
REFERENCE_SHA = "966dcc405ba3d084d06912bdf8136da561e0473dfd119b1ed9f07d223dd9ed0b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neural", type=Path, required=True)
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    if args.run_dir.exists():
        p.error("Use a new output directory")
    if sha(args.neural) != REFERENCE_SHA:
        raise ValueError("Use the predeclared Stage-14 B average")
    device, _ = setup("cpu", "fp32", 4)
    base = torch.load(args.neural, map_location="cpu", weights_only=True)
    counts = torch.load(args.counts, map_location="cpu", weights_only=True)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if base["protocol"] != PROTOCOL or counts["protocol"] != PROTOCOL:
        raise ValueError("Unexpected protocol")
    provenance = counts["statistics_provenance"]
    for filename in ("tokenizer.json", "wikitext_validation.txt", "wikitext_train.txt"):
        if sha(ROOT / "data" / filename) != manifest["sha256"][filename]:
            raise ValueError("Changed data")
    if provenance["train_sha256"] != manifest["sha256"]["wikitext_train.txt"]:
        raise ValueError("Unexpected count training source")
    if (counts["config"]["max_order"], counts["config"]["min_count"], counts["config"]["discount"]) != (5,3,0.75):
        raise ValueError("Unexpected count recipe")
    raw = (ROOT / "data/wikitext_validation.txt").read_bytes()
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw.decode("utf8")).ids)
    neural, _ = make_model(base["implementation"], base["config"], device)
    neural.load_state_dict(base["model"])
    neural.eval()
    stats = build_model(counts["config"]).eval()
    stats.load_state_dict(counts["model"])
    args.run_dir.mkdir(parents=True)
    sources = {f: sha(ROOT / f) for f in ("student_ngram.py", "student.py", "student_structured.py",
               "scripts/screen_ngram_mixture.py", "scripts/build_ngram.py", "common.py", "evaluate.py")}
    totals = {str(w): 0.0 for w in WEIGHTS}
    stats_nll, targets = 0., 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch, (x,y) in enumerate(windows(ids, 32)):
            valid = y != -100
            safe_y = y.clamp_min(0).unsqueeze(-1)
            a = neural.predict_log_probs(x)
            b = stats.predict_log_probs(x)
            for output in (a,b):
                if not torch.isfinite(output).all() or output.logsumexp(-1).abs().max() > 1e-3:
                    raise ValueError("Invalid full probability distribution")
            a_target = a.gather(-1, safe_y).squeeze(-1)[valid]
            b_target = b.gather(-1, safe_y).squeeze(-1)[valid]
            stats_nll -= b_target.double().sum().item()
            targets += int(valid.sum())
            for w in WEIGHTS:
                mixed = a_target if w == 0 else torch.logaddexp(a_target + math.log1p(-w), b_target + math.log(w))
                totals[str(w)] -= mixed.double().sum().item()
            if batch % 10 == 0:
                print(json.dumps(dict(batch=batch, targets=targets)), flush=True)
    if targets != 376599 or len(raw) != 1148007:
        raise ValueError("Validation coverage mismatch")
    rows = [dict(weight=w, nll_nats=totals[str(w)], bpb=totals[str(w)] / math.log(2) / len(raw)) for w in WEIGHTS]
    best = min(rows, key=lambda r: r["bpb"])
    result = dict(protocol=PROTOCOL, split="validation", device="cpu", precision="fp32", targets=targets,
                  utf8_bytes=len(raw), reference_sha256=sha(args.neural), counts_sha256=sha(args.counts),
                  source_hashes=sources, count_provenance=provenance, candidates=rows,
                  standalone_ngram_bpb=stats_nll / math.log(2) / len(raw),
                  provisional_best=best, scan_seconds=time.perf_counter()-start,
                  note="Scan timing reuses neural outputs and is not a resource gate; no test accessed.")
    atomic_json_dump(result, args.run_dir / "scan.json")
    if best["weight"] > 0:
        config = dict(counts["config"], kind="hybrid", neural_config=base["config"], mixture_weight=best["weight"])
        hybrid = build_model(config).eval()
        hybrid.neural.load_state_dict(base["model"])
        hybrid.ngram.load_state_dict(counts["model"])
        payload = checkpoint_payload(hybrid, "student_ngram", config, base["seed"], base["train_tokens"])
        payload["ancestry"] = dict(neural_sha256=sha(args.neural), counts_sha256=sha(args.counts),
                                   neural_training_targets=base["train_tokens"], count_provenance=provenance)
        checkpoint = args.run_dir / "best-mixture.pt"
        atomic_torch_save(payload, checkpoint)
        output = args.run_dir / "best-validation-cpu-fp32.json"
        subprocess.run([sys.executable, "evaluate.py", "--checkpoint", str(checkpoint.resolve()),
                        "--device", "cpu", "--precision", "fp32", "--split", "validation", "--threads", "4",
                        "--output", str(output.resolve())], cwd=ROOT, check=True)
        official = json.loads(output.read_text())
        if abs(official["bpb"] - best["bpb"]) > 1e-6:
            raise ValueError("Scan and independently loaded scorer disagree")
        result["official_best_bpb"] = official["bpb"]
        result["best_checkpoint_sha256"] = sha(checkpoint)
        result["inference_asset_bytes"] = checkpoint.stat().st_size + sum((ROOT / f).stat().st_size for f in (
            "student_ngram.py", "student_structured.py", "student.py", "data/tokenizer.json", "common.py", "evaluate.py", "requirements.txt"))
        result["resource_gate_pending"] = True
    for filename, digest in sources.items():
        if sha(ROOT / filename) != digest:
            raise ValueError("Source changed during scan")
    result["status"] = "completed validation screen; resource qualification still required"
    atomic_json_dump(result, args.run_dir / "scan.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
