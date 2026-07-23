import unittest

import torch
from torch import nn

from capture_pipeline.exposure import HistogramRawAE, LearnedAEAdapter, RawExposureSynthesizer
from capture_pipeline.types import RawFrame


def make_frame(value=0.25, pattern="RGGB", requires_grad=False):
    mosaic = torch.full((1, 1, 4, 4), value, requires_grad=requires_grad)
    return RawFrame(mosaic, 0.0, 1.0, pattern, {"scene": "test"}, True)


class ExposureSynthesisTests(unittest.TestCase):
    def test_hard_exposure_doubles_and_halves(self):
        synth = RawExposureSynthesizer(ev_min=-4, ev_max=4, clipping_mode="hard")
        brighter, diag = synth(make_frame(0.25), torch.tensor([1.0]))
        darker, _ = synth(make_frame(0.25), torch.tensor([-1.0]))
        self.assertTrue(torch.allclose(brighter.mosaic, torch.full_like(brighter.mosaic, 0.5)))
        self.assertTrue(torch.allclose(darker.mosaic, torch.full_like(darker.mosaic, 0.125)))
        self.assertEqual(diag["exposure_scale"], [2.0])
        self.assertEqual(brighter.cfa_pattern, "RGGB")
        self.assertEqual(brighter.metadata, {"scene": "test"})

    def test_hard_clipping_and_per_cfa_diagnostics(self):
        frame = make_frame(0.75)
        exposed, diag = RawExposureSynthesizer(clipping_mode="hard")(frame, torch.tensor([1.0]))
        self.assertTrue(torch.equal(exposed.mosaic, torch.ones_like(exposed.mosaic)))
        self.assertEqual(diag["saturation_ratio"], [1.0])
        self.assertEqual(set(diag["per_cfa_saturation_ratio"][0]), {"R", "G1", "G2", "B"})
        self.assertTrue(all(v == 1.0 for v in diag["per_cfa_saturation_ratio"][0].values()))

    def test_soft_clipping_keeps_gradient_to_ev(self):
        frame = make_frame(0.8)
        ev = torch.tensor([1.0], requires_grad=True)
        exposed, _ = RawExposureSynthesizer(clipping_mode="soft", soft_beta=8.0)(frame, ev)
        exposed.mosaic.mean().backward()
        self.assertIsNotNone(ev.grad)
        self.assertTrue(torch.isfinite(ev.grad).all())
        self.assertGreater(abs(ev.grad.item()), 0.0)
        self.assertLessEqual(exposed.mosaic.max().item(), 1.0)

    def test_rejects_invalid_ev_shape_or_range(self):
        synth = RawExposureSynthesizer(ev_min=-2, ev_max=2)
        with self.assertRaisesRegex(ValueError, "shape"):
            synth(make_frame(), torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, "outside"):
            synth(make_frame(), torch.tensor([3.0]))


class AETests(unittest.TestCase):
    def test_histogram_ae_returns_batched_finite_outputs(self):
        raw = RawFrame(torch.rand(2, 1, 8, 8), 0.0, 1.0, "BGGR", {}, True)
        output = HistogramRawAE(target_gray=0.18, ev_min=-3, ev_max=3)(raw)
        self.assertEqual(tuple(output.ev.shape), (2,))
        self.assertEqual(tuple(output.confidence.shape), (2,))
        self.assertTrue(torch.isfinite(output.ev).all())
        self.assertTrue(((output.ev >= -3) & (output.ev <= 3)).all())
        self.assertTrue(((output.confidence >= 0) & (output.confidence <= 1)).all())

    def test_learned_adapter_accepts_tensor_and_dict(self):
        class TensorModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2, 3), keepdim=True)

        class DictModel(nn.Module):
            def forward(self, x):
                return {"ev": torch.ones(x.shape[0]), "confidence": torch.full((x.shape[0],), 0.7)}

        raw = RawFrame(torch.full((2, 1, 4, 4), 0.5), 0.0, 1.0, "RGGB", {}, True)
        tensor_out = LearnedAEAdapter(TensorModel(), ev_min=-2, ev_max=2)(raw)
        dict_out = LearnedAEAdapter(DictModel(), ev_min=-2, ev_max=2)(raw)
        self.assertEqual(tuple(tensor_out.ev.shape), (2,))
        self.assertTrue(torch.allclose(dict_out.ev, torch.ones(2)))
        self.assertTrue(torch.allclose(dict_out.confidence, torch.full((2,), 0.7)))


if __name__ == "__main__":
    unittest.main()
