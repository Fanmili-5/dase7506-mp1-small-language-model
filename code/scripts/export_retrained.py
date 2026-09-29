"""Export freshly trained experts with the fixed final mixture settings, without test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import torch
from common import PROTOCOL, make_model, setup, sha
from scripts.train_stage86_calibration_aware import train_log_prior
from train_experiment import atomic_torch_save, checkpoint_payload

FEATURES = ["neural_max_logp", "neural_margin", "highest_backoff", "highest_max_mass"]


def build_predictor(neural_payload, count_payload, gate, log_prior):
    if (neural_payload.get("protocol") != PROTOCOL
            or count_payload.get("protocol") != PROTOCOL
            or neural_payload.get("implementation") != "student_hybrid_conv_output_bias"
            or count_payload.get("implementation") != "student_ngram"
            or count_payload["config"].get("max_order") != 6
            or gate.get("no_validation_or_test_fitting") is not True):
        raise ValueError("Unexpected expert or gate protocol")
    config = dict(count_payload["config"], kind="gated_hybrid",
                  neural_config=neural_payload["config"], feature_names=FEATURES,
                  vocabulary_temperature=1.125, train_unigram_prior_weight=.0625,
                  copy_gate_shift=.25, anchor_weight=.0625, slope_scale=.5)
    model, _ = make_model("student_stage105_gated_singlepass", config, torch.device("cpu"))
    model.neural.load_state_dict(neural_payload["model"], strict=True)
    model.ngram.load_state_dict(count_payload["model"], strict=True)
    indices = [gate["feature_names"].index(name) for name in FEATURES]
    with torch.no_grad():
        model.log_prior.copy_(log_prior)
        model.gate_mean.copy_(torch.tensor([gate["feature_mean"][i] for i in indices],
                                          dtype=torch.float64))
        model.gate_std.copy_(torch.tensor([gate["feature_std"][i] for i in indices],
                                         dtype=torch.float64))
        model.gate_coeff.copy_(torch.tensor([gate["coefficients"][n] for n in FEATURES],
                                           dtype=torch.float64))
    if not torch.isfinite(model.gate_std).all() or (model.gate_std <= 0).any():
        raise ValueError("Invalid gate normalization")
    model.eval()
    reference, _ = make_model("student_stage103_gated", config, torch.device("cpu"))
    reference.load_state_dict(model.state_dict(), strict=True)
    reference.eval()
    ids = (torch.arange(512).reshape(2, 256) * 31 + 11) % 2048
    with torch.inference_mode():
        predicted = model.predict_log_probs(ids)
        expected = reference.predict_log_probs(ids)
        if not torch.isfinite(predicted).all():
            raise ValueError("Non-finite exported probabilities")
        torch.testing.assert_close(predicted, expected, atol=3e-5, rtol=1e-6)
        torch.testing.assert_close(predicted.logsumexp(-1), torch.zeros(2, 256),
                                   atol=1e-5, rtol=0)
        changed = ids.clone()
        changed[:, 128:] = (changed[:, 128:] + 7) % 2048
        torch.testing.assert_close(model(changed)[:, :128], predicted[:, :128],
                                   atol=3e-5, rtol=1e-6)
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("neural", "counts", "gate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    setup("cpu", "fp32", 4)
    gate = json.loads(args.gate.read_text(encoding="utf-8-sig"))
    if (gate["neural_sha256"] != sha(args.neural)
            or gate["extended_counts_sha256"] != sha(args.counts)):
        raise ValueError("Gate does not belong to these freshly trained experts")
    neural = torch.load(args.neural, map_location="cpu", weights_only=True)
    counts = torch.load(args.counts, map_location="cpu", weights_only=True)
    prior, _ = train_log_prior()
    model, config = build_predictor(neural, counts, gate, prior)
    payload = checkpoint_payload(model, "student_stage105_gated_singlepass", config,
                                 neural.get("seed", 92017), neural.get("train_tokens", 0))
    payload["retraining"] = dict(neural_sha256=sha(args.neural), counts_sha256=sha(args.counts),
                                gate_sha256=sha(args.gate), no_test_scoring=True,
                                resource_qualified=False, frozen_release=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, args.output)
    print(json.dumps({"checkpoint": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
