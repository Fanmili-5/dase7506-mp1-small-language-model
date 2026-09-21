"""Synthetic ancestry tests, never training/submission evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from common import PROTOCOL
from scripts.average_checkpoints import average_checkpoints
from scripts.audit_stage15_screen import audit, check_average


class Stage15AuditTests(unittest.TestCase):
    def test_refuses_unfinished_screen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "screen.json").write_text(json.dumps(dict(status="training", protocol=PROTOCOL, split="validation")))
            with self.assertRaisesRegex(ValueError, "must be completed"):
                audit(root)

    def test_recomputes_average_and_rejects_forged_weights_and_ancestry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "checkpoints").mkdir()
            paths = []
            for step in (6000, 6300, 6600, 6900, 7200):
                path = root / "checkpoints" / f"step-{step:06d}.pt"
                torch.save(dict(protocol=PROTOCOL, implementation="student", config={}, seed=17,
                    train_tokens=step * 8192, model={"weight": torch.tensor([step / 7200])}), path)
                paths.append(path)
            payload = average_checkpoints(paths)
            check_average(root, payload, 7200)
            payload["model"]["weight"].add_(.1)
            with self.assertRaisesRegex(ValueError, "Recomputed average"):
                check_average(root, payload, 7200)
            payload = average_checkpoints(paths)
            payload["averaging_ancestry"]["source_checkpoint_sha256"][0] = "bad"
            with self.assertRaisesRegex(ValueError, "ancestry"):
                check_average(root, payload, 7200)


if __name__ == "__main__":
    unittest.main()
