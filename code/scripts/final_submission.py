"""Freeze, verify and package the final execution of the unchanged Stage143 model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
EVIDENCE = CODE / 'results/final-evidence'
CHECKPOINT = 'checkpoints/final-model.pt'
IMPLEMENTATION = 'student_static_openvino_singlepass.py'
EXPECTED_CHECKPOINT = 'b8c1273f9ae629f7e6667370177ff4312ecf9aead5209abd4319a3bb62aa67a4'
EXPECTED_IMPLEMENTATION = '111e34302cf1095a18c57fc4d01f880ce1cc11e58316eac381d8ba38776c8c12'
sys.path.insert(0, str(CODE))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def commit() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def measured_files(checkpoint: Path) -> dict:
    previous = read(CODE / 'results/stage143-evidence/final.json')
    files = {}
    for name in previous['inference_files']:
        path = CODE / name
        if sha(path) != previous['source_hashes'][name]:
            raise ValueError(f'Changed original inference dependency: {name}')
        files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    if sha(CODE / IMPLEMENTATION) != EXPECTED_IMPLEMENTATION:
        raise ValueError('Execution implementation differs from measured candidate')
    if sha(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('Checkpoint differs from the Linux timing evidence')
    files[IMPLEMENTATION] = {'sha256': sha(CODE / IMPLEMENTATION),
                             'bytes': (CODE / IMPLEMENTATION).stat().st_size}
    files[CHECKPOINT] = {'sha256': sha(checkpoint), 'bytes': checkpoint.stat().st_size}
    if sum(row['bytes'] for row in files.values()) > 64 * 1024**2:
        raise ValueError('Inference assets exceed 64 MiB')
    return files


def freeze_and_score(output: Path) -> None:
    """One frozen full-test reproduction; no training or post-test selection."""
    from scripts.export_static_runtime import export
    from scripts.package_stage143_final_release import validate_window_nll

    if output.exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    checkpoint = output / 'checkpoint.pt'
    export(checkpoint)
    files = measured_files(checkpoint)
    resources = {}
    for name in ('resources-linux-attempt1.json', 'resources-linux-attempt2.json'):
        path = EVIDENCE / name
        result = read(path)
        if (result['candidate']['checkpoint_sha256'] != EXPECTED_CHECKPOINT
                or result['candidate']['implementation_sha256'] != EXPECTED_IMPLEMENTATION
                or result['threads'] != 4 or result['repeats'] != 3):
            raise ValueError('Timing evidence does not match this predictor')
        resources[name] = {'sha256': sha(path),
                           'time_ratio': result['candidate_to_baseline_time_ratio'],
                           'time_limit_met_on_measured_host': result['within_five_x_time_limit']}
    selected = read(EVIDENCE / 'resources-linux-attempt2.json')
    if not (selected['within_five_x_time_limit'] and selected['within_four_gib_peak_rss_limit']):
        raise ValueError('Selected measurement does not meet resource limits')
    freeze = {
        'status': 'method_frozen_before_test', 'protocol': '7506-mp1-wt2-v2',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': commit(), 'checkpoint': CHECKPOINT,
        'implementation': IMPLEMENTATION, 'inference_files': files,
        'conservative_inference_asset_bytes': sum(row['bytes'] for row in files.values()),
        'validation_bpb': selected['candidate']['bpb_runs'][0],
        'resource_evidence': resources,
        'timing_scope': 'Linux attempt 2 passes; attempt 1 fails. No universal-host guarantee.',
        'algorithm_changes': False, 'new_training_targets': 0,
        'original_test_result_used_for_selection': False,
    }
    freeze_path = output / 'freeze.json'
    write(freeze_path, freeze)
    print('Freeze saved before starting the unchanged course scorer.', flush=True)
    subprocess.run([sys.executable, str(CODE / 'evaluate.py'),
                    '--checkpoint', str(checkpoint), '--device', 'cpu',
                    '--precision', 'fp32', '--threads', '4', '--split', 'test',
                    '--output', str(output / 'test.json')], cwd=CODE, check=True)
    test = read(output / 'test.json')
    validate_test(test, freeze)
    validate_window_nll(test, output / 'test.window-nll.npy')
    write(output / 'reproduction.json', {
        'freeze_record_sha256': sha(freeze_path),
        'full_test_result_sha256': sha(output / 'test.json'),
        'full_test_window_nll_sha256': sha(output / 'test.window-nll.npy'),
        'full_test_window_count': 1674,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'full_test_cpu_fp32_bpb': test['bpb'],
        'source_commit': commit(), 'training_launched': False,
    })


def validate_test(test: dict, freeze: dict) -> None:
    files = freeze['inference_files']
    expected = {'protocol': '7506-mp1-wt2-v2', 'split': 'test', 'device': 'cpu',
                'precision': 'fp32', 'targets': 428405, 'utf8_bytes': 1292013,
                'checkpoint_sha256': files[CHECKPOINT]['sha256'],
                'implementation_sha256': files[IMPLEMENTATION]['sha256'],
                'evaluator_sha256': files['evaluate.py']['sha256'],
                'tokenizer_sha256': files['data/tokenizer.json']['sha256']}
    if any(test.get(key) != value for key, value in expected.items()):
        raise ValueError('Full-test record does not identify the frozen predictor')
    if (not math.isfinite(test['bpb']) or test['bpb'] <= 0
            or not math.isfinite(test['nll_nats'])
            or abs(test['bpb'] - test['nll_nats'] / math.log(2) / 1292013) > 1e-10):
        raise ValueError('Full-test BPB arithmetic failed')


def verify() -> dict:
    from scripts.package_stage143_final_release import validate_window_nll

    freeze, test = read(EVIDENCE / 'freeze.json'), read(EVIDENCE / 'test.json')
    reproduction = read(EVIDENCE / 'reproduction.json')
    if (freeze['status'] != 'method_frozen_before_test'
            or freeze['checkpoint'] != CHECKPOINT or freeze['implementation'] != IMPLEMENTATION):
        raise ValueError('Missing pre-test freeze')
    if measured_files(CODE / CHECKPOINT) != freeze['inference_files']:
        raise ValueError('Inference set differs from the freeze')
    for name, expected in freeze['resource_evidence'].items():
        if sha(EVIDENCE / name) != expected['sha256']:
            raise ValueError('Timing evidence changed')
    for name, key in (('freeze.json', 'freeze_record_sha256'),
                      ('test.json', 'full_test_result_sha256'),
                      ('test.window-nll.npy', 'full_test_window_nll_sha256')):
        if sha(EVIDENCE / name) != reproduction[key]:
            raise ValueError(f'Reproduction record changed: {name}')
    if (reproduction['source_commit'] != freeze['source_commit']
            or reproduction['full_test_cpu_fp32_bpb'] != test['bpb']
            or datetime.fromisoformat(reproduction['completed_at_utc'])
            <= datetime.fromisoformat(freeze['frozen_at_utc'])):
        raise ValueError('Reproduction chronology or score mismatch')
    validate_test(test, freeze)
    validate_window_nll(test, EVIDENCE / 'test.window-nll.npy')
    manifest_path = REPO / 'BUNDLE_MANIFEST.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if (manifest['files'] != freeze['inference_files']
                or manifest['report_sha256'] != sha(REPO / 'REPORT.pdf')
                or manifest['full_test_result_sha256'] != sha(EVIDENCE / 'test.json')
                or manifest['full_test_cpu_fp32_bpb'] != test['bpb']):
            raise ValueError('Bundle does not match the code/report/evidence')
    return {'frozen_files_verified': len(freeze['inference_files']),
            'inference_asset_bytes': freeze['conservative_inference_asset_bytes'],
            'recorded_test_bpb': test['bpb'], 'scored_by_this_check': False}


def package(output: Path) -> dict:
    from scripts.package_stage143_pretest_candidate import archived_entry

    verify()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO):
        raise ValueError('Commit the final code and report before packaging')
    freeze, test = read(EVIDENCE / 'freeze.json'), read(EVIDENCE / 'test.json')
    manifest = {
        'status': 'final_frozen_submission', 'protocol': freeze['protocol'],
        'release_code_commit': commit(), 'frozen_source_commit': freeze['source_commit'],
        'files': freeze['inference_files'],
        'conservative_inference_asset_bytes': freeze['conservative_inference_asset_bytes'],
        'report_sha256': sha(REPO / 'REPORT.pdf'),
        'freeze_record_sha256': sha(EVIDENCE / 'freeze.json'),
        'full_test_result_sha256': sha(EVIDENCE / 'test.json'),
        'full_test_window_nll_sha256': sha(EVIDENCE / 'test.window-nll.npy'),
        'full_test_cpu_fp32_bpb': test['bpb'],
        'resource_evidence': freeze['resource_evidence'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', allowZip64=True) as archive:
        for name, row in sorted(manifest['files'].items()):
            data = (CODE / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != row['sha256'] or len(data) != row['bytes']:
                raise ValueError(f'File changed during packaging: {name}')
            archived_entry(archive, 'code/' + name, data)
        archived_entry(archive, 'BUNDLE_MANIFEST.json',
                       (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
        archived_entry(archive, 'README.txt', (
            'Extract at the matching repository root. Follow code/README.md.\n'
            'From code/: python evaluate.py --checkpoint checkpoints/final-model.pt '
            '--device cpu --precision fp32 --threads 4 --split test '
            '--output reproduced-test.json\n').encode())
    with zipfile.ZipFile(output) as archive:
        expected = {'code/' + name for name in manifest['files']} | {'BUNDLE_MANIFEST.json', 'README.txt'}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError('Bundle members differ from manifest')
        for name, row in manifest['files'].items():
            if hashlib.sha256(archive.read('code/' + name)).hexdigest() != row['sha256']:
                raise ValueError('Bundle member verification failed')
    return {**manifest, 'zip_sha256': sha(output), 'zip_bytes': output.stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze-and-score', 'verify', 'package'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action != 'verify' and args.output is None:
        parser.error('--output is required')
    if args.action == 'freeze-and-score':
        freeze_and_score(args.output.resolve())
    else:
        print(json.dumps(verify() if args.action == 'verify' else package(args.output), indent=2))
