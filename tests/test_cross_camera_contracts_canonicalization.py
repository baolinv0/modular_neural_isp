import unittest

import torch

from cross_camera_tm.canonicalization import CanonicalizationConfig, DeviceCanonicalizer
from cross_camera_tm.contracts import (
    AlignmentQuality,
    CalibrationDataset,
    CalibrationPair,
    LinearMetadata,
    TargetSample,
    canonical_tensor_sha256,
)


def metadata(**overrides):
    payload = {
        "sample_id": "pair-001-iphone",
        "device": "iphone",
        "white_level": 10000.0,
        "is_normalized": False,
        "black_level_corrected": True,
        "white_balanced": True,
        "awb_gains_applied": [1.0, 2.0, 1.0],
        "reference_awb_gains": [2.0, 1.0, 1.0],
        "awb_gains_comparable": True,
        "ccm_to_common": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "exposure_time_s": 0.01,
        "iso": 100.0,
        "aperture": 2.0,
        "reference_exposure_product": 0.25,
        "hdr_confidence": 0.5,
        "metadata_complete": False,
    }
    payload.update(overrides)
    return LinearMetadata.from_mapping(payload)


class LinearMetadataContractTests(unittest.TestCase):
    def test_rejects_unknown_fields_and_invalid_upstream_state(self):
        valid = metadata()
        self.assertEqual(valid.sample_id, "pair-001-iphone")
        payload = valid.to_dict()
        payload["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "unknown fields.*surprise"):
            LinearMetadata.from_mapping(payload)
        with self.assertRaisesRegex(ValueError, "black-level-corrected"):
            metadata(black_level_corrected=False)
        with self.assertRaisesRegex(ValueError, "white-balanced"):
            metadata(white_balanced=False)

    def test_rejects_invalid_confidence_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "hdr_confidence"):
            metadata(hdr_confidence=1.1)
        with self.assertRaisesRegex(ValueError, "white_level"):
            metadata(white_level=float("nan"))
        with self.assertRaisesRegex(ValueError, "positive"):
            metadata(awb_gains_applied=[1.0, 0.0, 1.0])

    def test_canonical_tensor_hash_binds_shape_dtype_and_values(self):
        a = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
        b = a.clone()
        self.assertEqual(canonical_tensor_sha256(a), canonical_tensor_sha256(b))
        b[0, 0, 0, 0] = 1.0
        self.assertNotEqual(canonical_tensor_sha256(a), canonical_tensor_sha256(b))
        self.assertNotEqual(canonical_tensor_sha256(a), canonical_tensor_sha256(a.double()))

    def test_target_contract_rejects_unknown_fields_and_hash_mismatch(self):
        image = torch.full((1, 3, 4, 4), 0.2)
        md = metadata(
            sample_id="target-1",
            is_normalized=True,
            white_level=1.0,
            awb_gains_applied=[1.0, 1.0, 1.0],
            reference_awb_gains=[1.0, 1.0, 1.0],
        )
        payload = {
            "sample_id": "target-1",
            "scene_group": "scene-1",
            "split": "locked_holdout",
            "iphone_linear": image,
            "metadata": md,
            "input_sha256": canonical_tensor_sha256(image),
        }
        self.assertEqual(TargetSample.from_mapping(payload).split, "locked_holdout")
        with self.assertRaisesRegex(ValueError, "unknown target sample"):
            TargetSample.from_mapping({**payload, "unknown": 1})
        with self.assertRaisesRegex(ValueError, "canonical tensor bytes"):
            TargetSample.from_mapping({**payload, "input_sha256": "0" * 64})

    def test_frozen_calibration_dataset_requires_40_plus_10_pairs(self):
        image = torch.full((1, 3, 2, 2), 0.2)
        digest = canonical_tensor_sha256(image)
        iphone_md = metadata(
            sample_id="iphone",
            is_normalized=True,
            white_level=1.0,
            awb_gains_applied=[1.0, 1.0, 1.0],
            reference_awb_gains=[1.0, 1.0, 1.0],
        )
        samsung_md = metadata(
            sample_id="samsung",
            device="samsung",
            is_normalized=True,
            white_level=1.0,
            awb_gains_applied=[1.0, 1.0, 1.0],
            reference_awb_gains=[1.0, 1.0, 1.0],
        )
        pairs = tuple(
            CalibrationPair(
                pair_id=f"pair-{index}",
                scene_group=f"scene-{index % 5}",
                split="development" if index < 40 else "locked",
                alignment_quality=AlignmentQuality.SCENE_ONLY,
                iphone_linear=image,
                samsung_linear=image,
                samsung_gt=image,
                iphone_metadata=iphone_md,
                samsung_metadata=samsung_md,
                iphone_sha256=digest,
                samsung_sha256=digest,
                gt_sha256=digest,
            )
            for index in range(50)
        )
        self.assertEqual(len(CalibrationDataset(pairs).pairs), 50)
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            CalibrationDataset(pairs[:-1])


class DeviceCanonicalizationTests(unittest.TestCase):
    def test_normalizes_then_aligns_awb_then_ccm_then_exposure(self):
        image = torch.tensor([2000.0, 4000.0, 3000.0]).view(1, 3, 1, 1).expand(1, 3, 4, 4)
        md = metadata(
            ccm_to_common=[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            reference_exposure_product=0.5,
        )
        result = DeviceCanonicalizer(CanonicalizationConfig(exposure_scale_max=2.0)).canonicalize(image, md)
        # Normalize: [.2,.4,.3]; AWB ratio: [.4,.2,.3]; CCM: [.5,.2,.3]; exposure x2.
        expected = torch.tensor([1.0, 0.4, 0.6]).view(1, 3, 1, 1).expand_as(result.image)
        self.assertTrue(torch.allclose(result.image, expected, atol=1e-6))
        self.assertEqual(result.operations, ("white_level", "awb_gain_alignment", "common_ccm", "exposure_prior"))
        self.assertAlmostEqual(result.exposure_scale, 2.0)

    def test_missing_metadata_degrades_confidence_and_does_not_invent_awb(self):
        image = torch.full((1, 3, 8, 8), 0.2)
        complete = DeviceCanonicalizer().canonicalize(image, metadata(is_normalized=True, white_level=1.0, metadata_complete=True))
        incomplete = DeviceCanonicalizer().canonicalize(
            image,
            metadata(
                is_normalized=True,
                white_level=1.0,
                awb_gains_applied=None,
                reference_awb_gains=None,
                ccm_to_common=None,
                exposure_time_s=None,
                iso=None,
                aperture=None,
                reference_exposure_product=None,
                hdr_confidence=0.0,
                metadata_complete=False,
            ),
        )
        self.assertLess(incomplete.confidence.overall, complete.confidence.overall)
        self.assertNotIn("awb_gain_alignment", incomplete.operations)
        self.assertTrue(torch.allclose(incomplete.image, image))

    def test_reliable_and_highlight_masks_are_explicit(self):
        image = torch.tensor([[[[0.001, 0.2], [0.5, 0.99]]]]).repeat(1, 3, 1, 1)
        result = DeviceCanonicalizer(
            CanonicalizationConfig(reliable_dark_threshold=0.01, highlight_threshold=0.98)
        ).canonicalize(
            image,
            metadata(
                is_normalized=True,
                white_level=1.0,
                awb_gains_applied=[1.0, 1.0, 1.0],
                reference_awb_gains=[1.0, 1.0, 1.0],
            ),
        )
        self.assertEqual(result.reliable_mask.tolist(), [[[[False, True], [True, False]]]])
        self.assertEqual(result.highlight_valid_mask.tolist(), [[[[True, True], [True, False]]]])
        self.assertEqual(result.reliable_coverage, 0.5)

    def test_rejects_bad_image_contract(self):
        canonicalizer = DeviceCanonicalizer()
        with self.assertRaisesRegex(ValueError, "shape"):
            canonicalizer.canonicalize(torch.zeros(3, 4, 4), metadata())
        bad = torch.zeros(1, 3, 4, 4)
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            canonicalizer.canonicalize(bad, metadata())


if __name__ == "__main__":
    unittest.main()
