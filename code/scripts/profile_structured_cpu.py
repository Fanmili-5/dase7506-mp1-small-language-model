"""Read-only component microbenchmarks; not the official resource gate."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from torch.utils.benchmark import Timer
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, make_model, setup, sha, windows


def validate_implementation(checkpoint, impl_sha):
    # Native checkpoints identify the module but do not contain its source hash.
    if checkpoint["implementation"] != "student_structured" or impl_sha != "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18":
        raise ValueError("Requires the original pinned StructuredLM implementation")


def combine(model, hidden, vocabulary, copy):
    log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
    log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
    gate = model.copy_gate(hidden)
    return torch.logaddexp(F.logsigmoid(-gate) + vocabulary,
                          F.logsigmoid(gate) + log_copy)


def head(model, hidden, ids):
    return combine(model, hidden, F.log_softmax(model.head(hidden), -1),
                   model.copy_distribution(hidden, ids))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output must be new")
    device, _ = setup("cpu", "fp32", 4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != PROTOCOL or checkpoint["config"].get("output_kind") != "prefix_copy":
        raise ValueError("Requires the fixed-protocol prefix-copy architecture")
    model, impl_sha = make_model(checkpoint["implementation"], checkpoint["config"], device)
    validate_implementation(checkpoint, impl_sha)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ("wikitext_validation.txt", "tokenizer.json"):
        if sha(ROOT / "data" / name) != manifest["sha256"][name]:
            raise ValueError("Changed fixed data")
    raw = (ROOT / "data/wikitext_validation.txt").read_text(encoding="utf8")
    ids = torch.tensor(Tokenizer.from_file(str(ROOT / "data/tokenizer.json")).encode(raw).ids)
    x, _ = next(windows(ids, 32))
    measurements = {}
    with torch.inference_mode():
        hidden = model.features(x)
        vocabulary = F.log_softmax(model.head(hidden), -1)
        copy = model.copy_distribution(hidden, x)
        expected = model.predict_log_probs(x)
        actual = combine(model, hidden, vocabulary, copy)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        functions = {
            "backbone": lambda: model.features(x),
            "vocabulary_projection_logsoftmax": lambda: F.log_softmax(model.head(hidden), -1),
            "copy_attention_scatter": lambda: model.copy_distribution(hidden, x),
            "mixture_with_cached_distributions": lambda: combine(model, hidden, vocabulary, copy),
            "complete_head": lambda: head(model, hidden, x),
            "complete_forward": lambda: model.predict_log_probs(x),
        }
        for name, fn in functions.items():
            result = Timer(stmt="fn()", globals={"fn": fn}, num_threads=4).blocked_autorange(min_run_time=1.0)
            measurements[name] = dict(median_seconds=result.median, iqr_seconds=result.iqr,
                number_per_run=result.number_per_run, raw_seconds=result.raw_times)
            print(name, result.median, flush=True)
    output = dict(protocol=PROTOCOL, split="validation", precision="fp32", threads=4,
        checkpoint_sha256=sha(args.checkpoint), implementation_sha256=impl_sha,
        profiler_sha256=sha(__file__), batch_shape=list(x.shape), measurements=measurements,
        note="Read-only first-batch microbenchmarks with cached intermediates; times are not additive, not an official CPU gate; no test or training.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
