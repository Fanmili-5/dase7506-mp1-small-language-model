"""Reuse Stage20 contract tests plus direct collapsed-recurrence checks."""
import torch
import tests.test_ngram_fast as base
import student_ngram_collapsed


class CollapsedNgramTests(base.FastNgramTests):
    def pair(self, empty=False):
        old, previous = super().pair(empty)
        config = dict(kind="hybrid", mixture_weight=.1, vocab=2048, context=256,
            order_shapes=[(t.keys.numel(),t.mass.numel()) for t in previous.ngram.tables],
            neural_config=dict(vocab=2048, context=256, width=24, heads=4, depth=2,
                position="rope", norm="rmsnorm", activation="swiglu", mlp_ratio=2,
                bias=False, dropout=.1, output_kind="prefix_copy", copy_dim=8))
        new = student_ngram_collapsed.build_model(config).eval()
        new.load_state_dict(old.state_dict())
        return old, new

    def test_sparse_expansion_on_existing_distribution_and_repeated_ids(self):
        for empty in (False, True):
            old, new = self.pair(empty)
            for length in (1, 2, 3, 4, 7, 256):
                ids = torch.tensor(([4,7,4,9,7,4,6,4,2,7]*26)[:length]).repeat(3,1)
                initial = torch.rand(3, length, 2048) * .001
                before = {k:v.clone() for k,v in new.ngram.state_dict().items()}
                with torch.no_grad():
                    expected = initial + .1 * old.ngram.distribution(ids)
                    actual = new.ngram.add_into(initial.clone(), ids, .1)
                torch.testing.assert_close(actual, expected, atol=3e-8, rtol=2e-6)
                for key, value in new.ngram.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_rejects_noncontiguous_output(self):
        _, new = self.pair()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            new.ngram.add_into(torch.zeros(1,2048,3).transpose(1,2), torch.ones(1,3,dtype=torch.long), .1)
