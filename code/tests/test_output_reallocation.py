import json
from pathlib import Path
import unittest

import torch

import student_multi_token

ROOT = Path(__file__).resolve().parents[1]


class OutputReallocationTests(unittest.TestCase):
    def test_wider_residual_stream_uses_smaller_copy_projection(self):
        base_config = json.loads((ROOT / "configs/stage26_multi_token.json").read_text())
        wide_config = json.loads((ROOT / "configs/stage51_width288_copy32.json").read_text())
        base = student_multi_token.build_model(base_config).eval()
        wide = student_multi_token.build_model(wide_config).eval()
        base_block = sum(parameter.numel() for parameter in base.blocks[0].parameters())
        wide_block = sum(parameter.numel() for parameter in wide.blocks[0].parameters())
        self.assertLess(abs(wide_block / base_block - 1), .003)
        self.assertEqual(wide_config["width"], 288)
        self.assertEqual(wide.blocks[0].mlp.input.out_features // 2, 528)
        self.assertEqual(wide.copy_query.out_features, 32)
        self.assertLess(wide.copy_query.weight.numel() + wide.copy_key.weight.numel(),
                        base.copy_query.weight.numel() + base.copy_key.weight.numel())
        ids = torch.randint(2048, (2, 32))
        with torch.inference_mode():
            output = wide(ids)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.logsumexp(-1), torch.zeros(2, 32), atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
