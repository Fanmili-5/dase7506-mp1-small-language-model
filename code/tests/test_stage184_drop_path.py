import json
from pathlib import Path
import unittest

import torch

from common import make_model
from drop_path_pilot import attach_drop_path

ROOT = Path(__file__).resolve().parents[1]


class DummyBlock(torch.nn.Module):
    def forward(self, x):
        return x + 2


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([DummyBlock()])

    def forward(self, x):
        return self.blocks[0](x)


class DropPathTests(unittest.TestCase):
    def test_train_skips_whole_block_per_sample_and_eval_is_identity(self):
        model = DummyModel()
        before = dict(model.state_dict())
        attach_drop_path(model, 0.1)
        x = torch.zeros(512, 3, 4)
        model.train()
        torch.manual_seed(17)
        output = model(x)
        skipped = torch.isclose(output[:, 0, 0], torch.tensor(0.0))
        kept = torch.isclose(output[:, 0, 0], torch.tensor(2 / 0.9))
        self.assertTrue(skipped.any())
        self.assertTrue(kept.any())
        self.assertTrue((skipped | kept).all())
        self.assertTrue(torch.equal(output, output[:, :1, :1].expand_as(output)))
        model.eval()
        self.assertTrue(torch.equal(model(x), x + 2))
        self.assertEqual(dict(model.state_dict()), before)

    def test_actual_stage54_model_same_initial_weights_and_eval_output(self):
        control = json.loads((ROOT / "configs/stage54_hybrid_conv_rdrop.json").read_text())
        pilot = json.loads((ROOT / "configs/stage184_drop_path_hybrid_rdrop.json").read_text())
        self.assertEqual({k: v for k, v in pilot.items() if k != "drop_path_rate"}, control)
        torch.manual_seed(17)
        baseline, _ = make_model("student_hybrid_conv_rdrop", control, torch.device("cpu"))
        torch.manual_seed(17)
        experiment, _ = make_model("student_hybrid_conv_rdrop", pilot, torch.device("cpu"))
        attach_drop_path(experiment, pilot["drop_path_rate"])
        for name, weights in baseline.state_dict().items():
            self.assertTrue(torch.equal(weights, experiment.state_dict()[name]), name)
        baseline.eval(); experiment.eval()
        ids = torch.randint(0, 2048, (2, 16))
        with torch.no_grad():
            self.assertTrue(torch.equal(baseline(ids), experiment(ids)))


if __name__ == "__main__":
    unittest.main()
