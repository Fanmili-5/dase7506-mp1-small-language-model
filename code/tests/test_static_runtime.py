"""Portable unit tests of the fixed-shape adapter; real parity runs separately."""
from unittest import TestCase, mock

import numpy as np
import torch

import student_static_openvino_singlepass as runtime


class FakeRequest:
    def __init__(self):
        self.output = np.empty((8, 256, 288), dtype=np.float32)
        self.inputs = []

    def infer(self, values):
        ids = values['ids']
        assert ids.shape == (8, 256) and ids.dtype == np.int64
        self.inputs.append(ids.copy())
        self.output[:] = ids[:, :, None]
        return {'hidden': self.output}


class StaticRuntimeTests(TestCase):
    def head(self):
        request = FakeRequest()
        compiled = mock.Mock()
        properties = {'INFERENCE_PRECISION_HINT': runtime.ov.Type.f32,
                      'INFERENCE_NUM_THREADS': 1, 'NUM_STREAMS': 1,
                      'ENABLE_CPU_PINNING': False}
        compiled.get_property.side_effect = properties.__getitem__
        compiled.create_infer_request.return_value = request
        core = mock.Mock()
        core.compile_model.return_value = compiled
        with mock.patch.object(runtime.ov, 'Core', return_value=core):
            head = runtime.StaticGraphNeuralHead(dict(width=288, vocab=2048, copy_dim=64, context=256))
        core.read_model.return_value.reshape.assert_called_once_with({'ids': [8, 256]})
        return head, request

    def test_arbitrary_rows_short_windows_and_owned_outputs(self):
        head, request = self.head()
        ids = torch.arange(33*17).reshape(33, 17)
        result = head.features(ids)
        self.assertEqual(tuple(result.shape), (33, 17, 288))
        torch.testing.assert_close(result, ids.float()[:, :, None].expand(-1, -1, 288))
        self.assertEqual(len(request.inputs), 5)
        self.assertTrue(all((x[:, 17:] == 0).all() for x in request.inputs))
        self.assertTrue((request.inputs[-1][1:] == 0).all())
        # FakeRequest reuses its output allocation, so missing .copy() is caught.
        head.features(torch.zeros((1, 1), dtype=torch.long))
        torch.testing.assert_close(result, ids.float()[:, :, None].expand(-1, -1, 288))

    def test_noncontiguous_inputs(self):
        head, _ = self.head()
        ids = torch.arange(18*12).reshape(18, 12)[::2, ::2]
        self.assertFalse(ids.is_contiguous())
        torch.testing.assert_close(head.features(ids)[:, :, 0], ids.float())

    def test_invalid_inputs(self):
        head, _ = self.head()
        for ids in (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 12, dtype=torch.long),
                    torch.zeros(1, 257, dtype=torch.long), torch.zeros(1, 2), torch.zeros(12, dtype=torch.long)):
            with self.assertRaises(ValueError):
                head.features(ids)
