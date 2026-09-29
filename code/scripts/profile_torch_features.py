"""Compare original PyTorch kernels with static OpenVINO on identical weights.

All parameters are read from the frozen checkpoint and graph. No training,
label scoring, saved prediction cache or inference-asset rewrite is performed.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import openvino as ov
import torch
from tokenizers import Tokenizer

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from common import setup, windows
import student_hybrid_conv_output_bias
from student_static_openvino_singlepass import StaticGraphNeuralHead
from student_stage143_openvino_singlepass import GRAPH, GRAPH_SHA256


def recover_neural():
    checkpoint = CODE / 'checkpoints/stage143-openvino-order6.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3'
    assert hashlib.sha256(GRAPH.read_bytes()).hexdigest() == GRAPH_SHA256
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']['neural_config']
    neural = student_hybrid_conv_output_bias.build_model(config)
    state = {name.removeprefix('neural.'): value for name, value in payload['model'].items()
             if name.startswith('neural.')}
    graph = ov.Core().read_model(str(GRAPH))
    for node in graph.get_ordered_ops():
        name = node.friendly_name
        if node.get_type_name() == 'Constant' and name.startswith('neural.'):
            key = name.removeprefix('neural.')
            assert key not in state
            state[key] = torch.from_numpy(node.get_data().copy())
        elif node.get_type_name() == 'MatMul':
            weight = node.input_value(1).get_node()
            if weight.get_type_name() != 'Constant':
                continue
            assert name.startswith('/blocks.') and name.endswith('/MatMul')
            key = name.removeprefix('/').removesuffix('/MatMul').replace('/', '.') + '.weight'
            assert key not in state
            assert not node.get_transpose_a() and not node.get_transpose_b()
            state[key] = torch.from_numpy(weight.get_data().T.copy())
    assert state.keys() == neural.state_dict().keys(), (state.keys() ^ neural.state_dict().keys())
    assert torch.equal(state['head.weight'], state['token.weight'])
    neural.load_state_dict(state, strict=True)
    assert all(torch.equal(neural.state_dict()[key], value) for key, value in state.items())
    neural.eval()
    return neural, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recover-only', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    setup('cpu', 'fp32', args.threads)
    neural, config = recover_neural()
    if args.recover_only:
        print('Every neural state tensor recovered from the frozen assets; tied embedding verified.')
        return
    reference = StaticGraphNeuralHead(config)
    tokenizer = Tokenizer.from_file(str(CODE / 'data/tokenizer.json'))
    tokens = torch.tensor(tokenizer.encode((CODE / 'data/wikitext_validation.txt').read_text(encoding='utf-8')).ids)
    inputs = [ids for ids, _ in itertools.islice(windows(tokens), 4)]
    runners = {
        'openvino_static8': reference.features,
        'pytorch32': neural.features,
        'pytorch8': lambda ids: torch.cat([neural.features(block) for block in ids.split(8)]),
    }
    elapsed = defaultdict(list)
    errors = defaultdict(float)
    with torch.no_grad():
        expected = [reference.features(ids) for ids in inputs]
        for runner in runners.values():
            runner(inputs[0])
        for repeat in range(2):
            for index, ids in enumerate(inputs):
                names = list(runners)
                if (repeat + index) % 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter()
                    actual = runners[name](ids)
                    elapsed[name].append(time.perf_counter() - start)
                    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
                    errors[name] = max(errors[name], float((actual - expected[index]).abs().max()))
    report = dict(platform=platform.platform(), threads=args.threads,
                  torch_version=torch.__version__, openvino_version=ov.__version__,
                  seconds=dict(elapsed), median_seconds={k: statistics.median(v) for k, v in elapsed.items()},
                  max_hidden_error=dict(errors), weights_recovered_exactly=True,
                  graph_sha256=GRAPH_SHA256, scored_targets=0, scored_test=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert max(errors.values()) < 1e-3


if __name__ == '__main__':
    main()
