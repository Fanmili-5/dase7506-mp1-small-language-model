import json
from pathlib import Path
import unittest

import torch

import student_deep_supervision
import student_multi_token
import student_structured

ROOT = Path(__file__).resolve().parents[1]


class MultiTokenTests(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / "configs/stage26_multi_token.json").read_text(encoding="utf-8"))
        config.update(width=24, heads=4, depth=4, mlp_ratio=2, copy_dim=8,
                      deep_supervision_layers=[2, 3])
        return config

    def test_base_initialization_and_rng_are_unchanged(self):
        config = self.config()
        torch.manual_seed(91)
        multi = student_multi_token.build_model(config)
        after_multi = torch.get_rng_state()
        torch.manual_seed(91)
        base = student_deep_supervision.build_model(
            {key: value for key, value in config.items() if key not in student_multi_token.FUTURE_KEYS})
        after_base = torch.get_rng_state()
        multi_common = {key: value for key, value in multi.state_dict().items()
                        if not key.startswith(("future_norms.", "future_projections."))}
        self.assertEqual(multi_common.keys(), base.state_dict().keys())
        self.assertTrue(all(torch.equal(value, base.state_dict()[key])
                            for key, value in multi_common.items()))
        self.assertTrue(torch.equal(after_multi, after_base))

    def test_training_loss_and_future_gradients(self):
        model = student_multi_token.build_model(self.config()).train()
        ids = torch.tensor([[4, 7, 4, 9, 2], [3, 8, 3, 1, 6]])
        targets = torch.tensor([[7, 4, 9, 2, 5], [8, 3, 1, 6, 4]])
        future = torch.tensor([
            [[4, 9, 2, 5, 3], [3, 1, 6, 4, 2]],
            [[9, 2, 5, 3, 7], [1, 6, 4, 2, 8]],
        ])
        loss, parts = model.training_loss(ids, targets, future)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(float(value) > 0 for value in parts.values()))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.future_projections.parameters()))

    def test_export_is_exact_and_auxiliary_state_is_removed(self):
        config = self.config()
        source = student_multi_token.build_model(config).eval()
        deployed = student_structured.build_model(student_multi_token.inference_config(config)).eval()
        state = student_multi_token.inference_state(source.state_dict())
        deployed.load_state_dict(state, strict=True)
        self.assertFalse(any(key.startswith(("auxiliary_norms.", "future_norms.", "future_projections."))
                             for key in state))
        ids = torch.tensor([[4, 7, 4, 9, 2], [3, 8, 3, 1, 6]])
        with torch.inference_mode():
            torch.testing.assert_close(source(ids), deployed(ids), atol=0, rtol=0)

    def test_invalid_configs_rejected(self):
        for offsets, weight in (([], .2), ([2, 2], .2), ([1, 2], .2), ([3, 2], .2), ([2, 3], 0)):
            config = self.config()
            config["future_prediction_offsets"] = offsets
            config["future_prediction_weight"] = weight
            with self.assertRaises(ValueError):
                student_multi_token.build_model(config)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "CUDA BF16 is unavailable")
    def test_cuda_bf16_backward(self):
        model = student_multi_token.build_model(self.config()).cuda().train()
        ids = torch.randint(0, 2048, (2, 16), device="cuda")
        targets = torch.randint(0, 2048, (2, 16), device="cuda")
        future = torch.randint(0, 2048, (2, 2, 16), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = model.training_loss(ids, targets, future)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
