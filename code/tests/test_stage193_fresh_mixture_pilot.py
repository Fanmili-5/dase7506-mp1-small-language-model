"""Stage193 loss substitution and development-only data loading checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts import train_stage193_fresh_mixture_pilot as pilot


class FakeNeural(torch.nn.Module):
    rdrop_alpha = .5

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([.1, -.3, .4]))

    def single_training_pass(self, ids, targets, future_targets):
        logp = F.log_softmax(self.logits, dim=-1).expand(*ids.shape, -1)
        primary = F.nll_loss(logp.flatten(0, 1), targets.flatten())
        deep = self.logits.square().mean()
        future = self.logits.abs().mean()
        total = primary + .2 * deep + .2 * future
        return total, logp, {"deep": deep.detach(), "future": future.detach()}


class Stage193Tests(unittest.TestCase):
    def test_primary_replacement_keeps_auxiliary_gradients(self):
        model = FakeNeural()
        ids = torch.tensor([[0, 1]])
        targets = torch.tensor([[1, 2]])
        future = torch.zeros((2, 1, 2), dtype=torch.long)
        counts = torch.tensor([[.3, .2]])
        loss, parts = pilot.training_loss(model, ids, targets, future,
                                          counts, "mixture")
        neural = F.log_softmax(model.logits, dim=-1)
        expected_target = torch.logaddexp(
            neural[targets] + torch.log(torch.tensor(.9375)),
            counts.log() + torch.log(torch.tensor(.0625)),
        )
        expected = (-expected_target.mean() + .2 * model.logits.square().mean()
                    + .2 * model.logits.abs().mean())
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        self.assertTrue(torch.allclose(parts["primary"], -expected_target.mean()))
        loss.backward()
        self.assertTrue(torch.isfinite(model.logits.grad).all())
        self.assertGreater(float(model.logits.grad.abs().sum()), 0)

    def test_loader_needs_no_test_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            tokenizer = Tokenizer(models.WordLevel(
                vocab={"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"
            ))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(data / "tokenizer.json"))
            (data / "wikitext_train.txt").write_text("a b a", encoding="utf-8")
            (data / "wikitext_validation.txt").write_text("b a", encoding="utf-8")
            names = ("tokenizer.json", "wikitext_train.txt", "wikitext_validation.txt")
            digests = {
                name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                for name in names
            }
            # The manifest may name the test split, but this loader must not
            # require the file itself to exist or inspect its checksum.
            digests["wikitext_test.txt"] = "0" * 64
            (data / "manifest.json").write_text(json.dumps({
                "protocol": pilot.PROTOCOL, "sha256": digests,
            }), encoding="utf-8")
            with patch.object(pilot, "ROOT", root):
                loaded = pilot.load_train_validation()
            self.assertEqual(set(loaded), {"train", "validation"})
            self.assertEqual(loaded["train"][1], len(b"a b a"))


if __name__ == "__main__":
    unittest.main()
