"""Synthetic checks of the fresh-run export and command dependency plan."""
import importlib
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
import torch

from common import PROTOCOL
from scripts.reproduce_training import commands, CODE
from scripts.export_retrained import build_predictor, FEATURES
from scripts.build_kneser_ney import fit_kneser_ney
from scripts.build_stage73_order6 import extend_model
from tests import test_hybrid_conv_output_bias as bias_tests
import student_hybrid_conv_output_bias


class ReproductionTests(unittest.TestCase):
    def test_plan_scripts_exist_and_do_not_score_test(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / "new-run"
            plan = commands(directory)
            self.assertEqual(len(plan), 26)
            self.assertFalse(directory.exists())
            for command in plan:
                self.assertTrue(Path(command[1]).is_file(), command[1])
                self.assertNotIn("test", command)
                self.assertNotIn("evaluate.py", command[1])
                sys.path.insert(0, str(CODE / "scripts"))
                try:
                    module = importlib.import_module("scripts." + Path(command[1]).stem)
                finally:
                    sys.path.pop(0)
                for relative in getattr(module, "SOURCE_FILES", ()):
                    self.assertTrue((CODE / relative).is_file(), relative)

    def test_fresh_predictor_is_causal_and_matches_reference(self):
        torch.set_num_threads(2)
        config = bias_tests.HybridConvOutputBiasTests.config()
        neural = student_hybrid_conv_output_bias.build_model(config)
        neural_payload = dict(protocol=PROTOCOL, implementation="student_hybrid_conv_output_bias",
                              config=config, model=neural.state_dict())
        ids = np.array([0, 1, 2, 3, 4, 5, 2, 1, 0, 3] * 10, dtype=np.int64)
        base, base_config, _ = fit_kneser_ney(ids, max_order=5, min_count=2)
        counts, count_config, _ = extend_model(
            dict(protocol=PROTOCOL, implementation="student_ngram",
                 config=base_config, model=base.state_dict()), ids)
        count_payload = dict(protocol=PROTOCOL, implementation="student_ngram",
                             config=count_config, model=counts.state_dict())
        gate = dict(no_validation_or_test_fitting=True, feature_names=FEATURES,
                    feature_mean=[0.] * 4, feature_std=[1.] * 4,
                    coefficients=dict(zip(FEATURES, [.1, -.1, .2, -.2])))
        model, _ = build_predictor(neural_payload, count_payload, gate,
                                  torch.full((2048,), -float(np.log(2048))))
        with torch.inference_mode():
            x = torch.arange(16).reshape(2, 8)
            torch.testing.assert_close(model(x)[:1], model(x[:1]), atol=3e-5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
