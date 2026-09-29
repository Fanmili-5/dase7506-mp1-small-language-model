"""The Stage179 inference architecture and initialization match Stage54."""
import json
from pathlib import Path
import unittest

import torch

import student_hybrid_conv_rdrop


ROOT = Path(__file__).resolve().parents[1]


class Stage179DropoutTests(unittest.TestCase):
    def test_only_main_dropout_changes(self):
        old = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        new = json.loads((ROOT / "configs/stage179_hybrid_dropout20_rdrop.json").read_text())
        self.assertEqual(old["dropout"], 0.1)
        self.assertEqual(new["dropout"], 0.2)
        self.assertEqual({k: v for k, v in old.items() if k != "dropout"},
                         {k: v for k, v in new.items() if k != "dropout"})

    def test_same_seed_initial_state_and_eval_prediction(self):
        old = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        new = json.loads((ROOT / "configs/stage179_hybrid_dropout20_rdrop.json").read_text())
        torch.manual_seed(17)
        control = student_hybrid_conv_rdrop.build_model(old).eval()
        torch.manual_seed(17)
        candidate = student_hybrid_conv_rdrop.build_model(new).eval()
        for name, tensor in control.state_dict().items():
            self.assertTrue(torch.equal(tensor, candidate.state_dict()[name]), name)
        ids = (torch.arange(32).reshape(2, 16) * 31 + 11) % 2048
        with torch.inference_mode():
            torch.testing.assert_close(control(ids), candidate(ids), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
