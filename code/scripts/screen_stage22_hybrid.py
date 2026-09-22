"""Scan a fixed count mixture grid for the exported Stage22 neural candidate."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer

from common import PROTOCOL, make_model, setup, sha, windows
from student_ngram import build_model
from train_experiment import atomic_json_dump, atomic_torch_save, checkpoint_payload

COUNTS_SHA = "b1898559c73ccf62e6e945c1a9269e6fe2b0ee4499b5a1d88e7e30020c194230"
WEIGHTS = (0., .05, .075, .10, .125, .15)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neural",type=Path,required=True)
    p.add_argument("--counts",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args()
    if args.run_dir.exists(): p.error("Use a new output directory")
    if sha(args.counts) != COUNTS_SHA: raise ValueError("Unexpected train counts")
    neural_payload=torch.load(args.neural,map_location="cpu",weights_only=True)
    count_payload=torch.load(args.counts,map_location="cpu",weights_only=True)
    if (neural_payload["protocol"] != PROTOCOL or neural_payload["implementation"] != "student_structured"
            or neural_payload.get("seed") != 17 or neural_payload.get("train_tokens") != 58982400
            or "training_deep_supervision" not in neural_payload):
        raise ValueError("Expected the exported Stage22 neural candidate")
    device,_=setup("cpu","fp32",4)
    neural,_=make_model("student_structured",neural_payload["config"],device)
    counts,_=make_model(count_payload["implementation"],count_payload["config"],device)
    neural.load_state_dict(neural_payload["model"]); neural.eval()
    counts.load_state_dict(count_payload["model"]); counts.eval()
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    for name in ("wikitext_validation.txt","tokenizer.json","wikitext_train.txt"):
        if sha(ROOT/"data"/name) != manifest["sha256"][name]: raise ValueError("Fixed data changed")
    raw=(ROOT/"data/wikitext_validation.txt").read_bytes()
    ids=torch.tensor(Tokenizer.from_file(str(ROOT/"data/tokenizer.json")).encode(raw.decode("utf8")).ids)
    totals={w:0. for w in WEIGHTS}; targets=0; started=time.perf_counter()
    with torch.inference_mode():
        for batch,(x,y) in enumerate(windows(ids,32)):
            valid=y!=-100; target=y.clamp_min(0).unsqueeze(-1)
            a=neural.predict_log_probs(x).gather(-1,target).squeeze(-1)[valid]
            b=counts.predict_log_probs(x).gather(-1,target).squeeze(-1)[valid]
            for w in WEIGHTS:
                mixed=a if w==0 else torch.logaddexp(a+math.log1p(-w),b+math.log(w))
                totals[w]-=mixed.double().sum().item()
            targets+=int(valid.sum())
            if batch%10==0: print(json.dumps(dict(batch=batch,targets=targets)),flush=True)
    rows=[dict(weight=w,nll_nats=totals[w],bpb=totals[w]/math.log(2)/len(raw)) for w in WEIGHTS]
    best=min(rows,key=lambda row:row["bpb"])
    if targets!=376599 or len(raw)!=1148007: raise ValueError("Coverage mismatch")
    args.run_dir.mkdir(parents=True)
    result=dict(protocol=PROTOCOL,split="validation",neural_sha256=sha(args.neural),
        counts_sha256=COUNTS_SHA,candidates=rows,best=best,targets=targets,utf8_bytes=len(raw),
        seconds=time.perf_counter()-started,new_gradient_targets=0,
        note="Fixed grid selected on validation; all statistics remain train-only. No test scoring.")
    if best["weight"]>0:
        config=dict(count_payload["config"],kind="hybrid",neural_config=neural_payload["config"],mixture_weight=best["weight"])
        hybrid=build_model(config).eval()
        hybrid.neural.load_state_dict(neural_payload["model"])
        hybrid.ngram.load_state_dict(count_payload["model"])
        payload=checkpoint_payload(hybrid,"student_ngram",config,17,58982400)
        payload["ancestry"]=dict(neural_sha256=sha(args.neural),counts_sha256=COUNTS_SHA,
            neural_training_targets=58982400,training_deep_supervision=neural_payload["training_deep_supervision"],
            count_provenance=count_payload["statistics_provenance"])
        checkpoint=args.run_dir/"best-mixture.pt"; atomic_torch_save(payload,checkpoint)
        output=args.run_dir/"best-validation-cpu-fp32.json"
        subprocess.run([sys.executable,"evaluate.py","--checkpoint",str(checkpoint.resolve()),
            "--device","cpu","--precision","fp32","--threads","4","--split","validation",
            "--output",str(output.resolve())],cwd=ROOT,check=True)
        official=json.loads(output.read_text())
        if abs(official["bpb"]-best["bpb"])>1e-6: raise ValueError("Scan/official mismatch")
        result.update(checkpoint_sha256=sha(checkpoint),official_bpb=official["bpb"])
    atomic_json_dump(result,args.run_dir/"scan.json")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
