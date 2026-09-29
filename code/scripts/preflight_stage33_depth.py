"""Serialize and resource-test the fixed random-weight Stage33 inference graph."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from common import PROTOCOL, make_model, sha
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload

COUNTS_SHA = "b91b42f069276a881e21eface99e285a40b2484e25e54d44176414886a20d2d2"
BASELINE_SHA = "2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d"
INFERENCE_FILES = (
    "student_ngram_collapsed.py", "student_ngram_fast.py", "student_ngram.py",
    "student_structured.py", "student.py", "common.py", "evaluate.py",
    "data/tokenizer.json", "requirements.txt",
)
TRAINING_KEYS = {
    "embedding_row_dropout", "ffn_hidden_dropout", "deep_supervision_layers",
    "deep_supervision_weight", "future_prediction_offsets", "future_prediction_weight",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage33_depth10_width224.json"))
    parser.add_argument("--architecture", default="10x224 standard Transformer, 7 heads, prefix-copy64, collapsed MKN .125")
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error("Use a new run directory")
    if sha(args.counts) != COUNTS_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError("Unexpected fixed counts or baseline checkpoint")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    training_config = json.loads(config_path.read_text())
    neural_config = {key: value for key, value in training_config.items() if key not in TRAINING_KEYS}
    count_payload = torch.load(args.counts, map_location="cpu", weights_only=True)
    if count_payload["protocol"] != PROTOCOL or count_payload["implementation"] != "student_ngram":
        raise ValueError("Unexpected count checkpoint")
    torch.manual_seed(17)
    neural, _ = make_model("student_structured", neural_config, torch.device("cpu"))
    config = dict(count_payload["config"], kind="hybrid", neural_config=neural_config,
                  mixture_weight=.125)
    candidate, _ = make_model("student_ngram_collapsed", config, torch.device("cpu"))
    candidate.neural.load_state_dict(neural.state_dict())
    candidate.ngram.load_state_dict(count_payload["model"])
    candidate.eval()
    args.run_dir.mkdir(parents=True)
    checkpoint = args.run_dir / "random-preflight.pt"
    payload = checkpoint_payload(candidate, "student_ngram_collapsed", config, 17, 0)
    payload["preflight"] = dict(random_weights=True, training_targets=0,
                                purpose="weight-independent Stage33 inference resource gate")
    atomic_torch_save(payload, checkpoint)
    resource_path = args.run_dir / "resources.json"
    subprocess.run([
        sys.executable, "scripts/benchmark_cpu.py", "--baseline", str(args.baseline.resolve()),
        "--candidate", str(checkpoint.resolve()), "--repeats", "3", "--threads", "4",
        "--output", str(resource_path.resolve()),
    ], cwd=ROOT, check=True)
    resources = json.loads(resource_path.read_text())
    assets = checkpoint.stat().st_size + sum((ROOT / name).stat().st_size for name in INFERENCE_FILES)
    result = dict(
        protocol=PROTOCOL, status="preflight_completed", seed=17,
        architecture=args.architecture, config_sha256=sha(config_path),
        neural_parameters=sum(parameter.numel() for parameter in neural.parameters()),
        candidate_checkpoint_sha256=sha(checkpoint), count_checkpoint_sha256=COUNTS_SHA,
        baseline_checkpoint_sha256=BASELINE_SHA, conservative_asset_bytes=assets,
        cpu_ratio=resources["candidate_to_baseline_time_ratio"],
        peak_rss_bytes=resources["candidate"]["max_peak_rss_bytes"],
        within_five_x_time_limit=resources["within_five_x_time_limit"],
        within_four_gib_peak_rss_limit=resources["within_four_gib_peak_rss_limit"],
        within_64mib_asset_limit=assets <= 64 * 1024 ** 2,
        no_quality_claim=True, new_gradient_targets=0, no_test_scoring=True,
    )
    result["pass"] = all((result["within_five_x_time_limit"],
                           result["within_four_gib_peak_rss_limit"],
                           result["within_64mib_asset_limit"]))
    atomic_json_dump(result, args.run_dir / "preflight.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
