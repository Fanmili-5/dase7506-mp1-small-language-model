"""Check runtime-only equivalence, causality and row independence on synthetic inputs."""
import argparse
import json
from pathlib import Path
import platform
import sys

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from common import make_model, setup, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device, _ = setup('cpu', 'fp32', args.threads)
    candidate = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # Both execution paths use the final tensors; no archived checkpoint is needed.
    reference, _ = make_model('student_stage143_openvino_singlepass', candidate['config'], device)
    model, _ = make_model(candidate['implementation'], candidate['config'], device)
    reference.load_state_dict(candidate['model'], strict=True)
    model.load_state_dict(candidate['model'], strict=True)
    reference.eval(); model.eval()
    generator = torch.Generator().manual_seed(240929)
    ids = torch.randint(0, 128, (10, 256), generator=generator)
    with torch.no_grad():
        expected = reference.predict_log_probs(ids)
        actual = model.predict_log_probs(ids)
        assert torch.isfinite(actual).all()
        probability_error = float((expected.exp()-actual.exp()).abs().max())
        logp_error = float((expected-actual).abs().max())
        normalization_error = float(actual.logsumexp(-1).abs().max())
        changed = ids.clone()
        changed[:, 91:] = (changed[:, 91:]+37) % 2048
        future_error = float((actual[:, :91]-model.predict_log_probs(changed)[:, :91]).abs().max())
        row_error = float((actual[:1]-model.predict_log_probs(ids[:1])).abs().max())
        short_error = float((actual[:2, :91]-model.predict_log_probs(ids[:2, :91])).abs().max())
        model.predict_log_probs(torch.randint(2048, (9, 13), generator=generator))
        repeated_error = float((actual-model.predict_log_probs(ids)).abs().max())
    checks = dict(max_probability_error=probability_error, max_logp_error=logp_error,
                  max_log_normalization_error=normalization_error,
                  max_causal_prefix_error=future_error, max_independent_row_error=row_error,
                  max_short_window_error=short_error, max_cross_call_state_error=repeated_error)
    passed = (probability_error <= 3e-6 and logp_error <= 1e-4 and normalization_error <= 2e-6
              and max(future_error, row_error, short_error, repeated_error) <= 1e-5)
    report = dict(checks=checks, passed=passed, checkpoint_sha256=sha(args.checkpoint),
                  implementation_sha256=sha(CODE/'student_static_openvino_singlepass.py'),
                  platform=platform.platform(), requested_threads=args.threads,
                  reference='dynamic OpenVINO execution with the same final checkpoint tensors',
                  execution_threads=model.neural.execution_threads,
                  training_launched=False, scored_test=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    assert passed, 'Runtime equivalence/causality check failed'


if __name__ == '__main__':
    main()
