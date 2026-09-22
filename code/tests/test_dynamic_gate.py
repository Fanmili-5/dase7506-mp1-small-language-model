import unittest
import numpy as np
import torch
from scripts.build_ngram import fit_tables
import student_dynamic_gate


class DynamicGateTests(unittest.TestCase):
    def config_state(self):
        counts,config,_=fit_tables(np.tile([4,7,4,9,7,4,6,4,2,7],8),min_count=3)
        config.update(kind="hybrid_dynamic_gate",mixture_weight=.1,initial_count_weight=.1,
            neural_config=dict(vocab=2048,context=256,width=24,heads=4,depth=2,position="rope",
                norm="rmsnorm",activation="swiglu",mlp_ratio=2,bias=False,dropout=.1,
                output_kind="prefix_copy",copy_dim=8))
        model=student_dynamic_gate.build_model(config)
        model.ngram.load_state_dict(counts.state_dict())
        return model

    def test_initial_gate_is_point_one_and_distribution_is_valid(self):
        model=self.config_state().eval(); ids=torch.tensor([[4,7,4,9,7]])
        with torch.inference_mode():
            hidden=model.neural.features(ids).float(); weight=model.dynamic_weight(hidden)
            torch.testing.assert_close(weight,torch.full_like(weight,.1),atol=2e-7,rtol=0)
            output=model(ids)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output.logsumexp(-1),torch.zeros(1,5),atol=2e-6,rtol=0)

    def test_only_input_prefix_controls_gate_and_gradients(self):
        model=self.config_state().eval(); ids=torch.tensor([[4,7,4,9,7],[3,8,3,1,6]])
        altered=ids.clone(); altered[:,3:]+=20
        output=model(ids); loss=-output[...,4].mean(); loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.count_gate.parameters()))
        with torch.inference_mode():
            torch.testing.assert_close(model(ids)[:,:3],model(altered)[:,:3],atol=3e-6,rtol=0)

    def test_nonfinite_gate_rejected(self):
        model=self.config_state(); state=model.state_dict(); state["count_gate.bias"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError,"finite"):
            model.load_state_dict(state)


if __name__=="__main__": unittest.main()
