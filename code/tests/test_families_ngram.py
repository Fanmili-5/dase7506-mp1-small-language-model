"""New-family and full-distribution count-model contracts; synthetic data only."""
import io
import unittest
import numpy as np
import torch
from torch.nn import functional as F

import student_families
import student_ngram
from scripts.build_ngram import fit_tables


class FamiliesNgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def family_config(self, family):
        return dict(vocab=2048, context=256, family=family, width=24, depth=2,
                    hidden=32, dropout=0.1, kernel=3, dilations=[1, 4])

    def stats(self):
        return fit_tables(np.tile([4, 7, 4, 9, 7, 4, 6, 4, 2, 7], 8))

    def hybrid(self):
        counts, config, _ = self.stats()
        config = dict(config, kind="hybrid", mixture_weight=0.1, neural_config=dict(
            vocab=2048, context=256, width=24, heads=4, depth=2, position="rope",
            norm="rmsnorm", activation="swiglu", mlp_ratio=2.6666667, bias=False,
            dropout=0.1, output_kind="prefix_copy", copy_dim=8))
        model = student_ngram.build_model(config)
        model.ngram.load_state_dict(counts.state_dict())
        return model, config

    def check_contract(self, model):
        model.eval()
        x = torch.tensor([[4,7,4,9,7,4,6,4,2,7,4,3], [4,7,8,9,7,4,6,4,2,7,4,3]])
        changed = x.clone()
        changed[:, 6:] += 1
        with torch.no_grad():
            a = model.predict_log_probs(x)
            b = model.predict_log_probs(changed)
            short = model.predict_log_probs(x[:, :6])
            alone = model.predict_log_probs(x[:1])
            one = model.predict_log_probs(x[:, :1])
            again = model.predict_log_probs(x)
        self.assertEqual(a.shape, (2,12,2048))
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a.logsumexp(-1), torch.zeros(2,12), atol=2e-6, rtol=0)
        torch.testing.assert_close(one.logsumexp(-1), torch.zeros(2,1), atol=2e-6, rtol=0)
        torch.testing.assert_close(a[:,:6], b[:,:6], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a[:,:6], short, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a[:1], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a, again, atol=0, rtol=0)

    def test_family_contracts_and_gradients(self):
        for family in ("lstm", "gated_conv"):
            model = student_families.build_model(self.family_config(family))
            self.check_contract(model)
            model.train()
            ids = torch.randint(2048, (2, 19))
            F.cross_entropy(model(ids[:,:-1]).flatten(0,1), ids[:,1:].flatten()).backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_counts_and_hybrid_contracts(self):
        counts, _, _ = self.stats()
        hybrid, _ = self.hybrid()
        self.check_contract(counts)
        self.check_contract(hybrid)
        hybrid.train()
        ids = torch.tensor([[4,7,4,9,7,4]])
        loss = F.cross_entropy(hybrid(ids[:,:-1]).flatten(0,1), ids[:,1:].flatten())
        loss.backward()
        gradients = [p.grad for p in hybrid.neural.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertEqual(len(list(counts.parameters())), 0)

    def test_pruning_returns_removed_mass_to_lower_order(self):
        ids = np.array([4,7,4,7,4,9,4,7,4,8])
        model, _, _ = fit_tables(ids, max_order=2, min_count=3)
        # Context 4: total=5; count(7)=3 retained, 9 and 8 pruned.
        expected = model.unigram * (1 - (3-.75)/5)
        expected = expected.clone()
        expected[7] += (3-.75)/5
        torch.testing.assert_close(model.distribution(torch.tensor([[4]]))[0,0], expected)
        torch.testing.assert_close(model.distribution(torch.tensor([[2030]]))[0,0], model.unigram)

    def test_empty_tables_back_off_without_inventing_a_match(self):
        model, _, _ = fit_tables(np.arange(20), min_count=99)
        out = model.distribution(torch.tensor([[0,1,2,3,4]]))
        torch.testing.assert_close(out, model.unigram.expand(1,5,-1))

    def test_reload_and_exact_zero_mixture(self):
        model, config = self.hybrid()
        model.eval()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = student_ngram.build_model(config).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        ids = torch.tensor([[4,7,4,9,7]])
        with torch.no_grad():
            torch.testing.assert_close(model(ids), restored(ids), atol=0, rtol=0)
            restored.weight = 0
            torch.testing.assert_close(restored(ids), restored.neural(ids), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_family_autocast(self):
        for family in ("lstm", "gated_conv"):
            model = student_families.build_model(self.family_config(family)).cuda().train()
            ids = torch.randint(2048, (2, 33), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(ids[:,:-1])
                loss = F.cross_entropy(p.flatten(0,1), ids[:,1:].flatten())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
