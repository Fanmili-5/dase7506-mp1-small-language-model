"""Repackage identical tensors/config with a new execution module; no scoring."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from common import PROTOCOL, sha

SOURCE_SHA = '256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3'


def export(output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Refusing to overwrite a checkpoint')
    source = CODE / 'checkpoints/stage143-openvino-order6.pt'
    if sha(source) != SOURCE_SHA:
        raise ValueError('Frozen source checkpoint changed')
    original = torch.load(source, map_location='cpu', weights_only=True)
    if original['protocol'] != PROTOCOL:
        raise ValueError('Protocol mismatch')
    payload = copy.deepcopy(original)
    payload['implementation'] = 'student_static_openvino_singlepass'
    payload['runtime_only_repack'] = {
        'source_checkpoint_sha256': SOURCE_SHA,
        'source_implementation': original['implementation'],
        'method': 'fixed_eight_row_fp32_execution_of_identical_feature_graph',
        'new_training_targets': 0,
        'original_score_and_qualification_not_inherited': True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    loaded = torch.load(output, map_location='cpu', weights_only=True)
    assert loaded['config'] == original['config']
    assert loaded['model'].keys() == original['model'].keys()
    assert all(torch.equal(loaded['model'][k], v) for k, v in original['model'].items())
    report = dict(checkpoint=str(output), checkpoint_sha256=sha(output),
                  state_tensors_identical=True, config_identical=True,
                  scored_test=False, new_training_targets=0)
    output.with_suffix('.export.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(export(parser.parse_args().output), indent=2))
