"""Diagnostic upper bound for an input-only highest-match-order mixture gate.

Weights are selected on validation, so this script never exports a predictor.
It measures whether a train-fitted gate using the same legal feature is worth it.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from common import make_model,setup,sha,windows
from train_experiment import atomic_json_dump
from scripts.analyze_stage22_gate_ceiling import NEURAL_SHA,COUNTS_SHA,WEIGHTS


def highest_order(model,ids):
    batch,length=ids.shape
    positions=torch.arange(length).expand(batch,-1).flatten()
    result=torch.ones(ids.numel(),dtype=torch.long)
    for history,table in enumerate(model.tables,1):
        if history>length or table.keys.numel()==0: continue
        keys=torch.zeros_like(ids)
        for lag in range(history-1,-1,-1):
            shifted=ids if lag==0 else F.pad(ids[:,:-lag],(lag,0))
            keys=keys*2048+shifted
        keys=keys.flatten()
        locations=torch.searchsorted(table.keys,keys).clamp_max(table.keys.numel()-1)
        found=(table.keys[locations]==keys)&(positions>=history-1)
        result[found]=history+1
    return result.view_as(ids)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neural",type=Path,required=True); p.add_argument("--counts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    if args.output.exists(): p.error("Output must be new")
    if sha(args.neural)!=NEURAL_SHA or sha(args.counts)!=COUNTS_SHA: raise ValueError("Fixed inputs required")
    device,_=setup("cpu","fp32",4)
    np=torch.load(args.neural,map_location="cpu",weights_only=True)
    cp=torch.load(args.counts,map_location="cpu",weights_only=True)
    neural,_=make_model(np["implementation"],np["config"],device); neural.load_state_dict(np["model"]); neural.eval()
    counts,_=make_model(cp["implementation"],cp["config"],device); counts.load_state_dict(cp["model"]); counts.eval()
    raw=(ROOT/"data/wikitext_validation.txt").read_bytes()
    ids=torch.tensor(Tokenizer.from_file(str(ROOT/"data/tokenizer.json")).encode(raw.decode()).ids)
    totals={order:torch.zeros(len(WEIGHTS),dtype=torch.float64) for order in range(1,6)}
    group_targets={order:0 for order in range(1,6)}; targets=0
    with torch.inference_mode():
        for batch,(x,y) in enumerate(windows(ids,32)):
            valid=y!=-100; target=y.clamp_min(0).unsqueeze(-1)
            a=neural.predict_log_probs(x).gather(-1,target).squeeze(-1)
            b=counts.predict_log_probs(x).gather(-1,target).squeeze(-1)
            order=highest_order(counts,x)
            for group in range(1,6):
                mask=valid&(order==group); group_targets[group]+=int(mask.sum())
                av=a[mask].double(); bv=b[mask].double()
                for i,w in enumerate(WEIGHTS):
                    if w==0: mixed=av
                    elif w==1: mixed=bv
                    else: mixed=torch.logaddexp(av+math.log1p(-w),bv+math.log(w))
                    totals[group][i]-=mixed.sum()
            targets+=int(valid.sum())
            if batch%10==0: print(json.dumps(dict(batch=batch,targets=targets)),flush=True)
    groups=[]; total_nll=0.
    for group in range(1,6):
        i=int(totals[group].argmin()); total_nll+=float(totals[group][i])
        groups.append(dict(highest_match_order=group,targets=group_targets[group],best_weight=WEIGHTS[i],
            nll_nats=float(totals[group][i])))
    result=dict(split="validation",purpose="input_feature_ceiling_only",feature="highest_matched_ngram_order",
        groups=groups,group_selected_bpb=total_nll/math.log(2)/len(raw),targets=targets,utf8_bytes=len(raw),
        selected_on_validation_no_export=True,
        note="Feature is legal at inference, but weights are validation-fit diagnostics. Train-only calibration is required for any candidate.")
    atomic_json_dump(result,args.output); print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
