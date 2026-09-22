"""Fit a linear count gate on train-only held-out text, then attach full counts."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tokenizers import Tokenizer

from common import PROTOCOL,make_model,setup,sha,windows
from scripts.build_ngram import fit_tables
from train_experiment import atomic_json_dump,atomic_torch_save,checkpoint_payload

COUNTS_SHA="b1898559c73ccf62e6e945c1a9269e6fe2b0ee4499b5a1d88e7e30020c194230"
EPOCHS=50


def mean_nll(model,tokens,device):
    total=torch.zeros((),device=device); targets=0
    with torch.inference_mode():
        for x,y in windows(tokens,32):
            x,y=x.to(device),y.to(device); valid=y!=-100
            logp=model(x); losses=-logp.gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid]
            total=total+losses.sum(); targets+=int(valid.sum())
    return total/targets,targets


def train_epoch(model,tokens,device,optimizer):
    target_total=len(tokens)-1; nll=0.; observed=0
    optimizer.zero_grad(set_to_none=True)
    for x,y in windows(tokens,32):
        x,y=x.to(device),y.to(device); valid=y!=-100
        logp=model(x); losses=-logp.gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid]
        (losses.sum()/target_total).backward()
        nll+=float(losses.detach().sum()); observed+=int(valid.sum())
    anchor=(model.count_gate.bias-math.log(.1/.9)).square().mean()
    regularizer=1e-4*model.count_gate.weight.square().mean()+1e-3*anchor
    regularizer.backward(); optimizer.step()
    if observed!=target_total: raise ValueError("Calibration coverage mismatch")
    return nll/observed,observed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neural",type=Path,required=True); p.add_argument("--counts",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True); args=p.parse_args()
    if args.run_dir.exists(): p.error("Use a new run directory")
    if sha(args.counts)!=COUNTS_SHA: raise ValueError("Use the fixed full train counts")
    neural_payload=torch.load(args.neural,map_location="cpu",weights_only=True)
    full_counts=torch.load(args.counts,map_location="cpu",weights_only=True)
    if (neural_payload["protocol"]!=PROTOCOL or neural_payload["implementation"]!="student_structured"
            or neural_payload.get("seed")!=17 or "training_deep_supervision" not in neural_payload):
        raise ValueError("Expected the Stage22 exported neural model")
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    for name in ("wikitext_train.txt","tokenizer.json"):
        if sha(ROOT/"data"/name)!=manifest["sha256"][name]: raise ValueError("Training inputs changed")
    ids=Tokenizer.from_file(str(ROOT/"data/tokenizer.json")).encode(
        (ROOT/"data/wikitext_train.txt").read_text()).ids
    cut90=int(len(ids)*.90); cut95=int(len(ids)*.95)
    # Count expert never observes either gate-fit or gate-selection segment.
    fold_model,fold_config,fold_summary=fit_tables(ids[:cut90],max_order=5,min_count=3,discount=.75)
    fold_config.update(kind="hybrid_dynamic_gate",mixture_weight=.1,initial_count_weight=.1,
        neural_config=neural_payload["config"])
    device,_=setup("cuda","fp32",4)
    model,_=make_model("student_dynamic_gate",fold_config,device)
    model.neural.load_state_dict(neural_payload["model"])
    model.ngram.load_state_dict(fold_model.state_dict())
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.count_gate.parameters(): parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.count_gate.parameters(),lr=.03,weight_decay=.01)
    train_tokens=torch.tensor(ids[cut90:cut95]); selection_tokens=torch.tensor(ids[cut95:])
    args.run_dir.mkdir(parents=True)
    history=[]; best=None; stale=0; started=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        loss,fit_targets=train_epoch(model,train_tokens,device,optimizer)
        selection,selection_targets=mean_nll(model,selection_tokens,device)
        row=dict(epoch=epoch,fit_mean_nll=loss,selection_mean_nll=float(selection),
            mean_weight=float(model.dynamic_weight(model.neural.features(
                next(windows(selection_tokens,32))[0].to(device)).float()).mean()),
            fit_targets=fit_targets,selection_targets=selection_targets)
        history.append(row); print(json.dumps(row),flush=True)
        if best is None or row["selection_mean_nll"]<best["selection_mean_nll"]-1e-7:
            best=dict(row,gate={k:v.detach().cpu().clone() for k,v in model.count_gate.state_dict().items()}); stale=0
        else: stale+=1
        if stale>=8: break
    full_config=dict(full_counts["config"],kind="hybrid_dynamic_gate",mixture_weight=.1,
        initial_count_weight=.1,neural_config=neural_payload["config"])
    deployed,_=make_model("student_dynamic_gate",full_config,torch.device("cpu"))
    deployed.neural.load_state_dict(neural_payload["model"])
    deployed.ngram.load_state_dict(full_counts["model"])
    deployed.count_gate.load_state_dict(best.pop("gate"))
    payload=checkpoint_payload(deployed,"student_dynamic_gate",full_config,17,
        neural_payload.get("train_tokens",58982400))
    payload["ancestry"]=dict(neural_sha256=sha(args.neural),counts_sha256=COUNTS_SHA,
        neural_training_targets=neural_payload.get("train_tokens"),
        count_provenance=full_counts["statistics_provenance"])
    payload["gate_calibration"]=dict(method="linear_hidden_gate_train_only_90_5_5",
        train_token_count=len(ids),count_fit_end=cut90,gate_fit_end=cut95,
        gate_fit_unique_targets=len(train_tokens)-1,selection_unique_targets=len(selection_tokens)-1,
        epochs_completed=len(history),optimizer="AdamW lr=.03 wd=.01",early_stop_patience=8,
        best=best,fold_count_summary=fold_summary,no_validation_or_test_fitting=True,
        source_sha256=sha(Path(__file__)))
    checkpoint=args.run_dir/"dynamic-gate.pt"; atomic_torch_save(payload,checkpoint)
    report=dict(status="train_only_gate_fitted_validation_pending",checkpoint_sha256=sha(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,history=history,best=best,
        seconds=time.perf_counter()-started,calibration=payload["gate_calibration"],
        note="Neural expert saw the full supplied training text in its earlier training; count fold excludes both gate segments. No validation/test used here.")
    atomic_json_dump(report,args.run_dir/"fit.json"); print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
