import unittest

import torch

from common import PROTOCOL
from scripts.screen_stage28_weight_averages import soup_payload


class WeightAverageTests(unittest.TestCase):
    def payload(self, value):
        return dict(protocol=PROTOCOL, implementation="student_structured",
                    config={"vocab": 2048, "context": 256},
                    model={"weight": torch.full((2, 3), value),
                           "index": torch.tensor([1], dtype=torch.int64)})

    def test_interpolation_and_metadata(self):
        result = soup_payload(self.payload(4.), self.payload(0.), .75)
        torch.testing.assert_close(result["model"]["weight"], torch.full((2, 3), 3.))
        torch.testing.assert_close(result["model"]["index"], torch.tensor([1]))
        self.assertEqual(result["weight_soup"]["stage22_weight"], .75)
        self.assertEqual(result["weight_soup"]["new_gradient_targets"], 0)

    def test_incompatible_sources_rejected(self):
        right = self.payload(0.)
        right["config"] = {"vocab": 2048, "context": 255}
        with self.assertRaises(ValueError):
            soup_payload(self.payload(1.), right, .5)


if __name__ == "__main__":
    unittest.main()
