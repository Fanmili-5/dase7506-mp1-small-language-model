import json
from pathlib import Path
import unittest

import torch
import student_deep_supervision
import student_structured

ROOT = Path(__file__).resolve().parents[1]


class DeepSupervisionTests(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / "configs/stage22_deep_supervision.json").read_text())
        config.update(width=24, heads=4, depth=4, mlp_ratio=2, copy_dim=8,
                      deep_supervision_layers=[2,3])
        return config

    def test_eval_export_state_and_output_are_exact(self):
        config = self.config()
        model = student_deep_supervision.build_model(config).eval()
        inference = student_structured.build_model(student_deep_supervision.inference_config(config)).eval()
        state = student_deep_supervision.inference_state(model.state_dict())
        inference.load_state_dict(state, strict=True)
        self.assertTrue(all(not key.startswith("auxiliary_norms.") for key in state))
        ids = torch.tensor([[4,7,4,9,2], [3,8,3,1,6]])
        with torch.inference_mode():
            torch.testing.assert_close(model(ids), inference(ids), atol=0, rtol=0)

    def test_training_loss_is_causal_finite_and_updates_auxiliary_norms(self):
        model = student_deep_supervision.build_model(self.config()).train()
        ids = torch.tensor([[4,7,4,9,2], [3,8,3,1,6]])
        targets = torch.tensor([[7,4,9,2,5], [8,3,1,6,4]])
        torch.manual_seed(11)
        loss, parts = model.training_loss(ids, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["primary"]), 0)
        self.assertGreater(float(parts["auxiliary"]), 0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.auxiliary_norms.parameters()))
        model.eval()
        altered = ids.clone(); altered[:,3:] += 20
        with torch.inference_mode():
            torch.testing.assert_close(model(ids)[:,:3], model(altered)[:,:3], atol=2e-6, rtol=0)

    def test_invalid_configs_rejected(self):
        for layers, weight in (([],.2), ([2,2],.2), ([4],.2), ([2],0)):
            config = self.config(); config["deep_supervision_layers"] = layers
            config["deep_supervision_weight"] = weight
            with self.assertRaises(ValueError):
                student_deep_supervision.build_model(config)


if __name__ == "__main__":
    unittest.main()
