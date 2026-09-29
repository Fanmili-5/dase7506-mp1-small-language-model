"""Audit Stage20 equivalence receipts, unchanged tensors and raw resource runs."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import sha, PROTOCOL
from scripts.prepare_stage20_fast import REFERENCE_SHA, INFERENCE_FILES
from scripts.audit_stage19_results import score, resource, close


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("Audit output must be new")
    def read(name):
        return json.loads((args.evidence/name).read_text(encoding="utf-8-sig"))
    status=read("job-logs/stage20-20260922-a/status.json")
    assert status["status"] == "completed" and status["exit_code"] == 0
    parity=read("equivalence.json")
    digest=sha(args.candidate)
    assert sha(args.source) == REFERENCE_SHA == parity["reference_sha256"]
    assert parity["checkpoint_sha256"] == digest and parity["protocol"] == PROTOCOL
    assert parity["split"] == "validation" and parity["device"] == "cpu" and parity["precision"] == "fp32"
    assert parity["targets"] == 376599 and parity["utf8_bytes"] == 1148007
    assert parity["new_gradient_targets"] == 0 and parity["unchanged_serialized_tensors"]
    assert parity["max_abs_logp_error"] <= 2e-5
    assert parity["max_abs_log_normalization_error"] <= 2e-6
    assert abs(parity["optimized_bpb"]-parity["old_bpb"]) <= 1e-6
    for name,expected in parity["source_hashes"].items():
        assert sha(ROOT/name) == expected, name
    source=torch.load(args.source,map_location="cpu",weights_only=True)
    candidate=torch.load(args.candidate,map_location="cpu",weights_only=True)
    assert candidate["implementation"] == "student_ngram_fast"
    for key in ("protocol","config","seed","train_tokens","ancestry"):
        assert source[key] == candidate[key],key
    assert set(source["model"]) == set(candidate["model"])
    for key,value in source["model"].items():
        assert torch.equal(value,candidate["model"][key]),key
    assert candidate["inference_optimization"]["source_checkpoint_sha256"] == REFERENCE_SHA
    assert candidate["inference_optimization"]["source_hashes"] == parity["source_hashes"]
    from student_ngram_fast import build_model
    rebuilt=build_model(candidate["config"])
    rebuilt.load_state_dict(candidate["model"])
    assert rebuilt.count_probability_lower_bound == parity["positive_count_floor"]
    official=read("validation-cpu-fp32.json")
    score(official,digest)
    close(official["bpb"],parity["optimized_bpb"])
    assert official["implementation_sha256"] == sha(ROOT/"student_ngram_fast.py")
    raw=read("resources.json")
    checked=resource(raw,digest,official["bpb"])
    for row in raw["candidate"]["runs"]:
        assert row["implementation_sha256"] == official["implementation_sha256"]
        assert row["checkpoint_bytes"] == args.candidate.stat().st_size
    assets=args.candidate.stat().st_size + sum((ROOT/f).stat().st_size for f in INFERENCE_FILES)
    assert assets == parity["asset_bytes"]
    checked.update(asset_bytes=assets,asset_pass=assets <= 64*1024**2)
    checked["qualified_on_measured_windows_cpu"] = checked["cpu_pass"] and checked["ram_pass"] and checked["asset_pass"]
    report=dict(status="verified",protocol=PROTOCOL,split="validation",bpb=official["bpb"],
        checkpoint_sha256=digest,reference_sha256=REFERENCE_SHA,resources=checked,
        new_gradient_targets=0,unchanged_tensors_and_training_ancestry=True,
        max_abs_logp_error=parity["max_abs_logp_error"],
        file_hashes={str(f.relative_to(args.evidence)):sha(f) for f in sorted(args.evidence.rglob("*.json"))},
        limitations="Numerical equivalence within preregistered tolerance, not bit-exact outputs. Timing applies to measured Windows CPU, not all machines. No test score or final release.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
