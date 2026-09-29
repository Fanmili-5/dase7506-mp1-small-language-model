"""Synthetic equality, causality, numerical safety and mutation tests."""
import unittest
import numpy as np
import torch
from scripts.build_ngram import fit_tables
import student_ngram
import student_ngram_fast


class FastNgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def pair(self, empty=False):
        counts, config, _ = fit_tables(np.tile([4,7,4,9,7,4,6,4,2,7], 8), min_count=9999 if empty else 3)
        config.update(kind="hybrid", mixture_weight=.1, neural_config=dict(
            vocab=2048, context=256, width=24, heads=4, depth=2, position="rope",
            norm="rmsnorm", activation="swiglu", mlp_ratio=2, bias=False,
            dropout=.1, output_kind="prefix_copy", copy_dim=8))
        old = student_ngram.build_model(config).eval()
        old.ngram.load_state_dict(counts.state_dict())
        new = student_ngram_fast.build_model(config).eval()
        new.load_state_dict(old.state_dict())
        return old, new

    def test_full_context_equality_and_no_buffer_mutation(self):
        old, new = self.pair()
        ids = torch.tensor(np.tile([4,7,4,9,7,4,6,4], 64).reshape(2,256))
        before = {k:v.clone() for k,v in new.state_dict().items()}
        with torch.no_grad():
            torch.testing.assert_close(old.ngram.distribution(ids), new.ngram.distribution(ids), rtol=0, atol=0)
            a, b = old(ids), new(ids)
            torch.testing.assert_close(a, b, rtol=0, atol=4e-6)
            self.assertTrue(torch.isfinite(b).all())
            torch.testing.assert_close(b.logsumexp(-1), torch.zeros(2,256), atol=2e-6, rtol=0)
            torch.testing.assert_close(new(ids), b, atol=0, rtol=0)
        for key, value in new.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_causality_independence_empty_and_short_windows(self):
        for empty in (False, True):
            old, new = self.pair(empty)
            ids = torch.tensor([[4,7,4,9,7,4,6,4,2,7], [4,7,8,9,7,4,6,4,2,7]])
            altered = ids.clone()
            altered[:,5:] += 10
            with torch.no_grad():
                a = new(ids)
                torch.testing.assert_close(a, old(ids), atol=4e-6, rtol=0)
                torch.testing.assert_close(a[:,:5], new(altered)[:,:5], atol=0, rtol=0)
                torch.testing.assert_close(a[:,:5], new(ids[:,:5]), atol=4e-6, rtol=0)
                torch.testing.assert_close(a[:1], new(ids[:1]), atol=4e-6, rtol=0)
                torch.testing.assert_close(new(ids[:,:1]), old(ids[:,:1]), atol=4e-6, rtol=0)

    def test_extreme_gate_and_logits_have_positive_normalized_support(self):
        old, new = self.pair()
        with torch.no_grad():
            old.neural.head.weight.mul_(100)
            old.neural.copy_gate.weight.zero_()
            for bias in (-1000., 1000.):
                old.neural.copy_gate.bias.fill_(bias)
                new.load_state_dict(old.state_dict())
                ids = torch.tensor([[4,7,4,9]])
                a,b = old(ids),new(ids)
                self.assertTrue(torch.isfinite(b).all())
                torch.testing.assert_close(a,b,atol=5e-6,rtol=0)
                torch.testing.assert_close(b.logsumexp(-1),torch.zeros(1,4),atol=2e-6,rtol=0)

    def test_training_path_preserved_and_floor_guard(self):
        old,new = self.pair()
        old.train(); new.train()
        ids = torch.tensor([[4,7,4,9,7]])
        torch.manual_seed(77)
        a=old(ids)
        torch.manual_seed(77)
        b=new(ids)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        (-b[...,4].mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.neural.parameters()))
        state = old.state_dict()
        state["ngram.unigram"] = torch.zeros_like(state["ngram.unigram"])
        with self.assertRaisesRegex(ValueError,"Unigram"):
            new.load_state_dict(state)

    def test_invalid_and_underflowing_statistics_are_rejected(self):
        old,new = self.pair()
        bad = {k:v.clone() for k,v in old.state_dict().items()}
        bad["ngram.tables.0.backoff"][0] = 0
        with self.assertRaisesRegex(ValueError,"backoff"):
            new.load_state_dict(bad)
        bad = {k:v.clone() for k,v in old.state_dict().items()}
        bad["ngram.unigram"][0] = 1e-40
        with self.assertRaisesRegex(ValueError,"floor too small"):
            new.load_state_dict(bad)


if __name__ == "__main__":
    unittest.main()
