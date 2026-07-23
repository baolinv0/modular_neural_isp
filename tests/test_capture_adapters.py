import tempfile
import unittest

import torch
from torch import nn

from capture_pipeline.adapters import (
    IdentityEnhancer,
    IdentityRawDenoiser,
    IdentityToneMapper,
    LinearColorTransform,
    MetadataAWB,
    ModuleAWBAdapter,
    ModuleDenoiserAdapter,
    ModuleEnhancerAdapter,
    ModuleToneAdapter,
    PhotofinishingToneAdapter,
    load_module_checkpoint,
)
from capture_pipeline.types import AWBOutput, RawFrame


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.frame = RawFrame(torch.full((1, 1, 4, 4), 0.25), 0.0, 1.0, "RGGB", {
            "illum_color": [0.5, 1.0, 0.8],
            "ccm": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }, True)
        self.rgb = torch.full((1, 3, 4, 4), 0.25, requires_grad=True)

    def test_identity_modules_preserve_tensor_and_gradient(self):
        denoised = IdentityRawDenoiser()(self.rgb, self.frame.metadata)
        tone = IdentityToneMapper()(denoised, {})
        enhanced = IdentityEnhancer()(tone.output)
        enhanced.mean().backward()
        self.assertTrue(torch.equal(enhanced, self.rgb))
        self.assertIsNotNone(self.rgb.grad)

    def test_metadata_awb_and_linear_color_transform(self):
        awb = MetadataAWB()(self.frame, self.rgb)
        self.assertEqual(tuple(awb.illuminant.shape), (1, 3))
        self.assertEqual(tuple(awb.ccm.shape), (1, 3, 3))
        output = LinearColorTransform()(self.rgb.detach(), awb)
        expected = torch.tensor([0.5, 0.25, 0.3125]).view(1, 3, 1, 1).expand_as(output)
        self.assertTrue(torch.allclose(output, expected))

    def test_metadata_awb_supports_alternative_keys_and_fails_missing(self):
        alt = RawFrame(self.frame.mosaic, 0.0, 1.0, "RGGB", {
            "cam_illum": [1.0, 1.0, 1.0],
            "color_matrix": torch.eye(3).tolist(),
        }, True)
        self.assertTrue(torch.allclose(MetadataAWB()(alt, self.rgb).illuminant, torch.ones(1, 3)))
        missing = RawFrame(self.frame.mosaic, 0.0, 1.0, "RGGB", {}, True)
        with self.assertRaisesRegex(KeyError, "illuminant"):
            MetadataAWB()(missing, self.rgb)

    def test_generic_module_adapters_and_awb_dictionary(self):
        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))
            def forward(self, x):
                return x * self.weight

        class AWBModel(nn.Module):
            def forward(self, x):
                return {"illuminant": torch.ones(x.shape[0], 3), "ccm": torch.eye(3).unsqueeze(0), "confidence": torch.ones(x.shape[0])}

        denoiser = ModuleDenoiserAdapter(Scale())
        enhancer = ModuleEnhancerAdapter(Scale())
        out = enhancer(denoiser(self.rgb, {}))
        out.mean().backward()
        self.assertGreater(denoiser.model.weight.grad.item(), 0)
        awb = ModuleAWBAdapter(AWBModel(), input_mode="rgb")(self.frame, self.rgb.detach())
        self.assertIsInstance(awb, AWBOutput)

    def test_generic_tone_adapter_accepts_tensor_or_mapping(self):
        class TensorTone(nn.Module):
            def forward(self, x):
                return x * 0.8

        class MappingTone(nn.Module):
            def forward(self, x):
                return {"output": x * 0.7, "gain": x * 0.9, "parameters": {"alpha": torch.tensor(0.7)}}

        tensor_tone = ModuleToneAdapter(TensorTone())(self.rgb.detach(), {})
        mapping_tone = ModuleToneAdapter(MappingTone())(self.rgb.detach(), {})
        self.assertTrue(torch.allclose(tensor_tone.output, self.rgb.detach() * 0.8))
        self.assertIn("gain", mapping_tone.stages)
        self.assertIn("alpha", mapping_tone.parameters)

    def test_photofinishing_adapter_extracts_intermediates(self):
        class FakePhotofinishing(nn.Module):
            def forward(self, x, return_intermediate=False, return_params=False):
                self.flags = (return_intermediate, return_params)
                return {
                    "output": x + 0.1,
                    "lsrgb_gain": x + 0.01,
                    "lsrgb_gtm": x + 0.02,
                    "lsrgb_ltm": x + 0.03,
                    "pred_gain": torch.tensor([[[[1.1]]]]),
                    "pred_gtm": torch.tensor([[1.0, 2.0, 3.0]]),
                    "pred_ltm": torch.ones(1, 4, 4, 4),
                }
        model = FakePhotofinishing()
        tone = PhotofinishingToneAdapter(model)(self.rgb.detach(), {})
        self.assertEqual(model.flags, (True, True))
        self.assertTrue(torch.allclose(tone.output, self.rgb.detach() + 0.1))
        self.assertEqual(set(tone.stages), {"gain", "gtm", "ltm"})
        self.assertEqual(tuple(tone.gtm.shape), (1, 3))

    def test_checkpoint_loading_preserves_exact_keys(self):
        module = nn.Linear(2, 1)
        expected = {k: torch.full_like(v, 0.5) for k, v in module.state_dict().items()}
        with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
            torch.save({"state_dict": expected}, handle.name)
            load_module_checkpoint(module, handle.name, strict=True)
        for value in module.state_dict().values():
            self.assertTrue(torch.allclose(value, torch.full_like(value, 0.5)))


if __name__ == "__main__":
    unittest.main()
