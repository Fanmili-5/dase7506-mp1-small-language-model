import json
from pathlib import Path
import unittest

import torch

import student_multi_token

ROOT = Path(__file__).resolve().parents[1]


class ComputeReallocationTests(unittest.TestCase):
    def test_projection_parameter_budget_is_matched(self):
        base_config = json.loads((ROOT / "configs/stage26_multi_token.json").read_text())
        wide_config = json.loads((ROOT / "configs/stage45_width288_mlp1833.json").read_text())
        base = student_multi_token.build_model(base_config).eval()
        wide = student_multi_token.build_model(wide_config).eval()
        base_block = base.blocks[0]
        wide_block = wide.blocks[0]
        base_weights = sum(p.numel() for p in base_block.parameters())
        wide_weights = sum(p.numel() for p in wide_block.parameters())
        self.assertLess(abs(wide_weights / base_weights - 1), .003)
        self.assertEqual(base_block.mlp.input.out_features // 2, 683)
        self.assertEqual(wide_block.mlp.input.out_features // 2, 528)
        ids = torch.randint(2048, (2, 32))
        with torch.no_grad():
            output = wide(ids)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.logsumexp(-1), torch.zeros(2, 32), atol=2e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
