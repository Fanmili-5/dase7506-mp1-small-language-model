"""Reject score, split and file mismatches in the final handoff."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import final_submission as final
from scripts.final_submission import EVIDENCE, read, validate_test, validate_window_nll, verify


class FinalSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.freeze = read(EVIDENCE / 'freeze.json')
        self.test = read(EVIDENCE / 'test.json')

    def test_recorded_final_files(self):
        result = verify()
        self.assertEqual(result['frozen_files_verified'], 19)
        self.assertEqual(result['inference_asset_bytes'], 55813469)
        self.assertEqual(result['recorded_test_bpb'], 1.4156576308174311)
        self.assertFalse(result['scored_by_this_check'])

    def test_rejects_validation_as_test(self):
        self.test['split'] = 'validation'
        with self.assertRaises(ValueError):
            validate_test(self.test, self.freeze)

    def test_rejects_wrong_identity_or_coverage(self):
        for key, value in (('checkpoint_sha256', '0' * 64),
                           ('implementation_sha256', '0' * 64),
                           ('evaluator_sha256', '0' * 64),
                           ('tokenizer_sha256', '0' * 64),
                           ('targets', 376599), ('utf8_bytes', 1148007),
                           ('precision', 'bf16'), ('device', 'cuda')):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.test)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_test(changed, self.freeze)

    def test_rejects_incorrect_score_arithmetic(self):
        for value in (1.399686, float('nan'), float('inf'), -1):
            with self.subTest(value=value):
                self.test['bpb'] = value
                with self.assertRaises(ValueError):
                    validate_test(self.test, self.freeze)

    def test_rejects_missing_inference_file(self):
        with tempfile.TemporaryDirectory() as scratch:
            with patch.object(final, 'CODE', Path(scratch)):
                with self.assertRaisesRegex(FileNotFoundError, 'Missing frozen file'):
                    final.measured_files(Path(scratch) / 'missing.pt')

    def test_rejects_changed_file_identity_or_size(self):
        for key, value in (('sha256', '0' * 64), ('bytes', 0)):
            changed = copy.deepcopy(self.freeze)
            changed['inference_files'][final.CHECKPOINT][key] = value
            with self.subTest(key=key), patch.object(final, 'read', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'Changed frozen inference file'):
                    final.measured_files(final.CODE / final.CHECKPOINT)

    def test_rejects_unsafe_manifest_paths(self):
        for name in ('../outside.py', '/absolute.py', 'folder\\outside.py'):
            changed = copy.deepcopy(self.freeze)
            changed['inference_files'] = {name: {'bytes': 0, 'sha256': '0' * 64}}
            with self.subTest(name=name), patch.object(final, 'read', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'Unsafe inference path'):
                    final.measured_files(final.CODE / final.CHECKPOINT)

    def test_window_sidecar_must_match_full_test_summary(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'window-nll.npy'
            losses = np.full(1674, self.test['nll_nats'] / 1674, dtype=np.float64)
            np.save(path, losses)
            self.assertEqual(validate_window_nll(self.test, path), 1674)
            losses[0] += 1
            np.save(path, losses)
            with self.assertRaisesRegex(ValueError, 'Window-NLL sum'):
                validate_window_nll(self.test, path)
            for invalid in (losses[:-1], losses.astype(np.float32),
                            np.full(1674, np.nan), np.full(1674, -1.0)):
                np.save(path, invalid)
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    validate_window_nll(self.test, path)


if __name__ == '__main__':
    unittest.main()
