import json
from pathlib import Path
import unittest

import torch

import student_rdrop_multi_token
import student_structured

ROOT = Path(__file__).resolve().parents[1]


class RDropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/stage47_rdrop.json").read_text())

    def test_loss_gradients_and_export(self):
        torch.manual_seed(17)
        model = student_rdrop_multi_token.build_model(self.config)
        ids = torch.randint(2048, (2, 24))
        targets = torch.randint(2048, ids.shape)
        future = torch.randint(2048, (2, *ids.shape))
        model.train()
        loss, parts = model.rdrop_training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(parts["symmetric_kl"]), -1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(model.token.weight.grad).all())
        deployed_config = student_rdrop_multi_token.inference_config(self.config)
        deployed = student_structured.build_model(deployed_config)
        deployed.load_state_dict(student_rdrop_multi_token.inference_state(model.state_dict()))
        model.eval(); deployed.eval()
        with torch.inference_mode():
            expected = model(ids)
            actual = deployed(ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros_like(actual[..., 0]),
                                   atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
