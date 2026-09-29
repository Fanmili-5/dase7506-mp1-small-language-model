import tempfile
import unittest
from pathlib import Path

import torch

from common import PROTOCOL
from scripts.average_checkpoints import average_checkpoints


class CheckpointAverageTests(unittest.TestCase):
    def test_same_trajectory_average_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, value in enumerate((1.0, 3.0), start=1):
                path = Path(directory) / f"step-{index}.pt"
                torch.save({
                    "protocol": PROTOCOL,
                    "implementation": "student",
                    "config": {"vocab": 2048, "context": 256},
                    "model": {"weight": torch.tensor([value, value + 2])},
                    "seed": 17,
                    "train_tokens": index * 100,
                }, path)
                paths.append(path)
            result = average_checkpoints(paths)
            torch.testing.assert_close(result["model"]["weight"], torch.tensor([2.0, 4.0]))
            self.assertEqual(result["train_tokens"], 200)
            self.assertEqual(result["averaging_ancestry"]["source_count"], 2)

    def test_rejects_independent_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (17, 23):
                path = Path(directory) / f"seed-{seed}.pt"
                torch.save({
                    "protocol": PROTOCOL,
                    "implementation": "student",
                    "config": {"vocab": 2048, "context": 256},
                    "model": {"weight": torch.tensor([1.0])},
                    "seed": seed,
                    "train_tokens": 100,
                }, path)
                paths.append(path)
            with self.assertRaises(ValueError):
                average_checkpoints(paths)


if __name__ == "__main__":
    unittest.main()
