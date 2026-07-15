import unittest

import torch
from torch import nn

from capture_pipeline.adapters import IdentityEnhancer, IdentityRawDenoiser, IdentityToneMapper, LinearColorTransform, MetadataAWB
from capture_pipeline.demosaic import BilinearBayerDemosaicer
from capture_pipeline.exposure import RawExposureSynthesizer
from capture_pipeline.pipeline import ModularCapturePipeline
from capture_pipeline.types import AEOutput, AWBOutput, RawFrame, ToneMapOutput


class RecordingAE(nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls
        self.bias = nn.Parameter(torch.tensor(0.0))
    def forward(self, raw):
        self.calls.append("ae")
        return AEOutput(self.bias.expand(raw.batch_size), torch.ones(raw.batch_size), {})


class RecordingExposure(nn.Module):
    def __init__(self, calls):
        super().__init__(); self.calls = calls
    def forward(self, raw, ev):
        self.calls.append("exposure")
        return raw.with_mosaic(raw.mosaic * torch.pow(torch.tensor(2.0, device=ev.device), ev).view(-1,1,1,1)), {"ok": True}


class RecordingDemosaic(nn.Module):
    def __init__(self, calls):
        super().__init__(); self.calls = calls
    def forward(self, raw):
        self.calls.append("demosaic")
        return raw.mosaic.repeat(1,3,1,1)


class RecordingDenoise(nn.Module):
    def __init__(self, calls):
        super().__init__(); self.calls = calls; self.weight = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, metadata):
        self.calls.append("denoise")
        return x * self.weight


class RecordingAWB(nn.Module):
    def __init__(self, calls):
        super().__init__(); self.calls = calls; self.bias = nn.Parameter(torch.tensor(0.0))
    def forward(self, raw, rgb):
        self.calls.append("awb")
        b = rgb.shape[0]
        illum = torch.ones(b,3,device=rgb.device,dtype=rgb.dtype) + self.bias * 0
        return AWBOutput(illum, torch.eye(3,device=rgb.device,dtype=rgb.dtype).unsqueeze(0).expand(b,3,3), torch.ones(b), {})


class RecordingColor(nn.Module):
    def __init__(self, calls):
        super().__init__(); self.calls = calls
    def forward(self, x, awb):
        self.calls.append("color")
        return x


class RecordingTone(nn.Module):
    def __init__(self, calls):
        super().__init__(); self.calls = calls; self.weight = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, context):
        self.calls.append("tone")
        y = x * self.weight
        return ToneMapOutput(y, None, None, None, {}, {"gtm": y})


class RecordingEnhance(nn.Module):
    def __init__(self, calls):
        super().__init__(); self.calls = calls; self.weight = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        self.calls.append("enhancement")
        return x * self.weight


def make_frame(value=0.2):
    return RawFrame(torch.full((1,1,6,6), value), 0.0, 1.0, "RGGB", {
        "illum_color": [1.0,1.0,1.0],
        "ccm": torch.eye(3).tolist(),
    }, True)


class PipelineTests(unittest.TestCase):
    def build_recording(self):
        calls = []
        pipe = ModularCapturePipeline(
            ae_estimator=RecordingAE(calls), exposure_synthesizer=RecordingExposure(calls),
            demosaicer=RecordingDemosaic(calls), denoiser=RecordingDenoise(calls),
            awb_estimator=RecordingAWB(calls), color_transform=RecordingColor(calls),
            tone_mapper=RecordingTone(calls), enhancer=RecordingEnhance(calls),
        )
        return pipe, calls

    def test_exact_module_order_and_stage_order(self):
        pipe, calls = self.build_recording()
        result = pipe(make_frame())
        self.assertEqual(calls, ["ae", "exposure", "demosaic", "denoise", "awb", "color", "tone", "enhancement"])
        self.assertEqual(list(result.stages), [
            "raw_input", "raw_normalized", "raw_exposed", "demosaiced_raw", "denoised_raw",
            "linear_awb", "tone_gtm", "tone_output", "enhanced_output", "final_srgb",
        ])
        self.assertTrue(torch.equal(result.final_srgb, result.stages["final_srgb"]))

    def test_overrides_bypass_ae_and_awb(self):
        pipe, calls = self.build_recording()
        result = pipe(
            make_frame(), override_ev=torch.tensor([1.0]),
            override_illuminant=torch.ones(1,3), override_ccm=torch.eye(3).unsqueeze(0),
        )
        self.assertNotIn("ae", calls)
        self.assertNotIn("awb", calls)
        self.assertTrue(torch.allclose(result.ae.ev, torch.tensor([1.0])))
        self.assertEqual(result.awb.diagnostics["source"], "override")

    def test_set_trainable_modules(self):
        pipe, _ = self.build_recording()
        pipe.set_trainable_modules(["tone", "enhancement"])
        state = pipe.module_trainability()
        self.assertFalse(state["ae"])
        self.assertFalse(state["denoiser"])
        self.assertFalse(state["awb"])
        self.assertTrue(state["tone"])
        self.assertTrue(state["enhancement"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            pipe.set_trainable_modules(["bad"])

    def test_joint_gradient_reaches_ae(self):
        class ScalarAE(nn.Module):
            def __init__(self):
                super().__init__(); self.ev = nn.Parameter(torch.tensor(0.0))
            def forward(self, raw):
                return AEOutput(self.ev.expand(raw.batch_size), torch.ones(raw.batch_size), {})

        ae = ScalarAE()
        pipe = ModularCapturePipeline(
            ae_estimator=ae,
            exposure_synthesizer=RawExposureSynthesizer(clipping_mode="soft", soft_beta=8.0),
            demosaicer=BilinearBayerDemosaicer(), denoiser=IdentityRawDenoiser(),
            awb_estimator=MetadataAWB(), color_transform=LinearColorTransform(),
            tone_mapper=IdentityToneMapper(), enhancer=IdentityEnhancer(),
        )
        output = pipe(make_frame(0.2))
        output.final_srgb.mean().backward()
        self.assertIsNotNone(ae.ev.grad)
        self.assertTrue(torch.isfinite(ae.ev.grad))
        self.assertGreater(abs(ae.ev.grad.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
