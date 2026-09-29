"""Compare dynamic and statically-shaped executions of one frozen FP32 graph."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import openvino as ov
import torch
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--code', type=Path, required=True)
p.add_argument('--threads', type=int, default=1)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
assert not args.output.exists()
sys.path.insert(0, str(args.code.resolve()))
from common import windows, setup
setup('cpu', 'fp32', args.threads)
graph = args.code / 'inference_assets/stage143-stage92-features.onnx'
assert hashlib.sha256(graph.read_bytes()).hexdigest() == '5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4'
tokenizer = Tokenizer.from_file(str(args.code / 'data/tokenizer.json'))
text = (args.code / 'data/wikitext_validation.txt').read_text(encoding='utf-8')
tokens = torch.tensor(tokenizer.encode(text).ids)
inputs = []
for ids, _ in windows(tokens):
    inputs.append(ids.numpy())
    if len(inputs) == 4:
        break
options = {'INFERENCE_PRECISION_HINT': ov.Type.f32,
           'INFERENCE_NUM_THREADS': args.threads,
           'NUM_STREAMS': 1, 'ENABLE_CPU_PINNING': False,
           'PERF_COUNT': True}
core = ov.Core()
requests = {}
for name, batch in [('dynamic', None), ('static32', 32), ('static8', 8)]:
    model = core.read_model(str(graph))
    if batch is not None:
        model.reshape({'ids': [batch, 256]})
    compiled = core.compile_model(model, 'CPU', options)
    assert compiled.get_property('INFERENCE_PRECISION_HINT') == ov.Type.f32
    assert int(compiled.get_property('INFERENCE_NUM_THREADS')) == args.threads
    requests[name] = compiled.create_infer_request()

def infer(name, ids):
    request = requests[name]
    chunk = 8 if name == 'static8' else 32
    outputs = []
    for offset in range(0, len(ids), chunk):
        outputs.append(next(iter(request.infer({'ids': ids[offset:offset+chunk]}).values())).copy())
    return np.concatenate(outputs)

reference = [infer('dynamic', ids) for ids in inputs]
for name in requests:
    infer(name, inputs[0])
seconds = defaultdict(list)
differences = defaultdict(float)
for repetition in range(2):
    for index, ids in enumerate(inputs):
        names = list(requests)
        if (index + repetition) % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            hidden = infer(name, ids)
            seconds[name].append(time.perf_counter()-start)
            assert hidden.dtype == np.float32 and np.isfinite(hidden).all()
            differences[name] = max(differences[name], float(np.abs(hidden-reference[index]).max()))
nodes = {}
for name, request in requests.items():
    types = defaultdict(float)
    for row in request.profiling_info:
        types[row.node_type + ':' + row.exec_type] += row.real_time.total_seconds()
    nodes[name] = dict(sorted(types.items(), key=lambda x:-x[1])[:12])
report = dict(platform=platform.platform(), threads=args.threads,
              openvino_version=ov.__version__, seconds=dict(seconds),
              median_seconds={name:float(np.median(rows)) for name, rows in seconds.items()},
              max_hidden_difference=dict(differences), top_node_types=nodes,
              graph_unchanged=True, no_test_scoring=True, scored_targets=0)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
assert max(differences.values()) < 1e-3
