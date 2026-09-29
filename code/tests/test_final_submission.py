"""Reject score, split and file mismatches in the final handoff."""
import copy
import unittest

from scripts.final_submission import EVIDENCE, read, validate_test, verify


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


if __name__ == '__main__':
    unittest.main()
