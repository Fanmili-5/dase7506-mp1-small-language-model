"""Measure feature-only versus whole-pipeline row tiling on validation inputs."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import torch
from tokenizers import Tokenizer

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from common import make_model, setup, windows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
device, _ = setup('cpu', 'fp32', 1)
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model, _ = make_model(payload['implementation'], payload['config'], device)
model.load_state_dict(payload['model']); model.eval()
tokenizer = Tokenizer.from_file(str(CODE/'data/tokenizer.json'))
tokens = torch.tensor(tokenizer.encode((CODE/'data/wikitext_validation.txt').read_text()).ids)
inputs = [ids for ids, _ in itertools.islice(windows(tokens), 4)]
component_times = {name: defaultdict(float) for name in ('features_only', 'whole_pipeline')}
active = 'features_only'

def instrument(call, label):
    def timed(*args, **kwargs):
        start = time.perf_counter()
        result = call(*args, **kwargs)
        component_times[active][label] += time.perf_counter()-start
        return result
    return timed

model.neural.features = instrument(model.neural.features, 'features')
model.ngram.collect = instrument(model.ngram.collect, 'count_collect')
model.ngram.add_collected = instrument(model.ngram.add_collected, 'count_add')

def predict(mode, ids):
    if mode == 'features_only':
        return model(ids)
    return torch.cat([model(block) for block in ids.split(8)], dim=0)

max_error = 0.0
max_probability_error = 0.0
normalization_errors = defaultdict(float)
seconds = defaultdict(list)
with torch.no_grad():
    for mode in component_times:
        predict(mode, inputs[0])
    for row in component_times.values():
        row.clear()
    for repetition in range(2):
        for index, ids in enumerate(inputs):
            order = ['features_only', 'whole_pipeline']
            if (index+repetition) % 2:
                order.reverse()
            outputs = {}
            for mode in order:
                active = mode
                start = time.perf_counter()
                outputs[mode] = predict(mode, ids)
                seconds[mode].append(time.perf_counter()-start)
            max_error = max(max_error, float((outputs['features_only']-outputs['whole_pipeline']).abs().max()))
            max_probability_error = max(max_probability_error, float((outputs['features_only'].exp()-outputs['whole_pipeline'].exp()).abs().max()))
            assert torch.isfinite(outputs['whole_pipeline']).all()
            for mode, output in outputs.items():
                normalization_errors[mode] = max(normalization_errors[mode], float(output.logsumexp(-1).abs().max()))
report = dict(platform=platform.platform(), threads=1, seconds=dict(seconds),
              component_times={k:dict(v) for k,v in component_times.items()},
              max_logp_error=max_error, max_probability_error=max_probability_error,
              normalization_errors=dict(normalization_errors),
              scored_test=False, scored_targets=0)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
assert max_error <= 1e-5
assert max_probability_error <= 3e-6
# Use the unchanged course scorer's normalization tolerance, recording both
# paths rather than aborting before diagnostics on an arbitrary tighter bound.
assert max(normalization_errors.values()) <= 1e-3
