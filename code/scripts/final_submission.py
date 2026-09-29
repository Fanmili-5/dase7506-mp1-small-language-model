"""Verify and package the frozen final coursework submission without scoring."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
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


def commit() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def measured_files(checkpoint: Path) -> dict:
    freeze = read(EVIDENCE / 'freeze.json')
    files = {}
    for name, expected in freeze['inference_files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError(f'Unsafe inference path: {name}')
        path = checkpoint if name == CHECKPOINT else CODE / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen file: {name}')
        actual = {'sha256': sha(path), 'bytes': path.stat().st_size}
        if actual != expected:
            raise ValueError(f'Changed frozen inference file: {name}')
        files[name] = actual
    if files[IMPLEMENTATION]['sha256'] != EXPECTED_IMPLEMENTATION:
        raise ValueError('Execution implementation differs from measured candidate')
    if files[CHECKPOINT]['sha256'] != EXPECTED_CHECKPOINT:
        raise ValueError('Checkpoint differs from the Linux timing evidence')
    total = sum(row['bytes'] for row in files.values())
    if total != freeze['conservative_inference_asset_bytes'] or total > 64 * 1024**2:
        raise ValueError('Incorrect inference byte count or assets exceed 64 MiB')
    return files


def validate_window_nll(test: dict, window_nll_path: Path) -> int:
    """Check the fixed scorer's ignored per-window sidecar against its summary."""
    import numpy as np

    if not window_nll_path.is_file():
        raise ValueError("Missing fixed-scorer window-NLL sidecar")
    losses = np.load(window_nll_path, allow_pickle=False)
    if (losses.shape != (1674,) or losses.dtype != np.float64
            or not np.isfinite(losses).all() or (losses < 0).any()):
        raise ValueError("Invalid complete-test window-NLL coverage")
    if abs(float(losses.sum(dtype=np.float64)) - test["nll_nats"]) > 1e-5:
        raise ValueError("Window-NLL sum differs from reported complete-test NLL")
    return int(losses.size)


def archived_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 26, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


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
    parser.add_argument('action', choices=('verify', 'package'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action != 'verify' and args.output is None:
        parser.error('--output is required')
    print(json.dumps(verify() if args.action == 'verify' else package(args.output), indent=2))
