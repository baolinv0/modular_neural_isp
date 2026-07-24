import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from cross_camera_tm.certification import (
    CRITICAL_GATE_NAMES,
    CertificationInputs,
    CertificationProfile,
    Certifier,
)
from cross_camera_tm.contracts import GateStatus, canonical_tensor_sha256
from cross_camera_tm.lineage import LineageNode, LineageStore
from cross_camera_tm.manifest import ManifestRecord, ManifestWriter
from cross_camera_tm.residuals import DirectionAlignmentResult
from cross_camera_tm.routing import (
    Route,
    StructuredUncertainty,
    SupervisionCandidate,
    SupervisionRouter,
)


def hash_of(text):
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def profile(*, synthetic=True):
    return CertificationProfile(
        profile_id="synthetic-canary" if synthetic else "real-calibration-v1",
        profile_sha256=hash_of("profile-s" if synthetic else "profile-r"),
        dataset_sha256=hash_of("dataset-s" if synthetic else "dataset-r"),
        synthetic=synthetic,
        sample_count=100,
        issue_lift_min=0.05,
        issue_lift_max=0.18,
        clipping_growth_max=0.01,
        geometry_correlation_min=0.98,
        high_frequency_correlation_min=0.85,
        high_frequency_energy_min=0.7,
        high_frequency_energy_max=1.4,
        chromaticity_mae_max=0.02,
        non_target_correction_max=0.04,
        boundary_artifact_max=0.03,
    )


def safe_inputs(**overrides):
    values = dict(
        phase1_valid=True,
        source_supported=True,
        eligible=True,
        input_supported=True,
        direction_alignment=DirectionAlignmentResult(GateStatus.PASS, 0.5, 4.0, ()),
        clipping_growth=0.0,
        geometry_correlation=0.999,
        high_frequency_correlation=0.96,
        high_frequency_energy_ratio=1.02,
        chromaticity_mae=0.0,
        non_target_correction=0.01,
        tm_feasible=True,
        boundary_artifact=0.0,
        candidate_kind="projected",
    )
    values.update(overrides)
    return CertificationInputs(**values)


class CertificationTests(unittest.TestCase):
    def setUp(self):
        self.reference = torch.full((1, 3, 16, 16), 0.2)
        self.candidate = torch.full((1, 3, 16, 16), 0.22)
        self.roi = torch.ones(1, 1, 16, 16)
        self.certifier = Certifier()

    def test_full_certification_has_every_critical_gate_and_passes(self):
        result = self.certifier.certify(
            self.reference, self.candidate, self.roi, safe_inputs(), profile()
        )
        self.assertEqual(tuple(gate.name for gate in result.gates), CRITICAL_GATE_NAMES)
        self.assertTrue(result.accepted)
        self.assertTrue(result.full_certification)

    def test_unavailable_cannot_pass_and_gates_do_not_compensate(self):
        unavailable = self.certifier.certify(
            self.reference,
            self.candidate,
            self.roi,
            safe_inputs(high_frequency_correlation=None),
            profile(),
        )
        direction_fail = self.certifier.certify(
            self.reference,
            self.candidate,
            self.roi,
            safe_inputs(direction_alignment=DirectionAlignmentResult(GateStatus.FAIL, 5.0, 4.0, ("sign",))),
            profile(),
        )
        boundary_fail = self.certifier.certify(
            self.reference, self.candidate, self.roi, safe_inputs(boundary_artifact=0.2), profile()
        )
        self.assertIs(unavailable.gate("HighFrequency").status, GateStatus.UNAVAILABLE)
        self.assertFalse(unavailable.accepted)
        self.assertIs(direction_fail.gate("DirectionAlignment").status, GateStatus.FAIL)
        self.assertFalse(direction_fail.accepted)
        self.assertIs(boundary_fail.gate("BoundaryArtifact").status, GateStatus.FAIL)

    def test_raw_generated_candidate_cannot_be_fully_accepted(self):
        result = self.certifier.certify(
            self.reference,
            self.candidate,
            self.roi,
            safe_inputs(candidate_kind="raw_generated"),
            profile(),
        )
        self.assertFalse(result.accepted)
        self.assertIs(result.gate("Eligibility").status, GateStatus.FAIL)

    def test_threshold_profile_is_derived_from_calibration_distribution(self):
        rows = [
            {
                "issue_lift": 0.05 + index * 0.001,
                "clipping_growth": index * 0.0001,
                "geometry_correlation": 0.99 - index * 0.0001,
                "high_frequency_correlation": 0.95 - index * 0.0002,
                "high_frequency_energy_ratio": 0.9 + index * 0.002,
                "chromaticity_mae": index * 0.0001,
                "non_target_correction": index * 0.0002,
                "boundary_artifact": index * 0.0001,
            }
            for index in range(40)
        ]
        calibrated = CertificationProfile.from_calibration(
            rows,
            profile_id="real-source-calibration",
            dataset_sha256=hash_of("real-source-dataset"),
            synthetic=False,
        )
        self.assertTrue(calibrated.real_calibrated)
        self.assertEqual(calibrated.sample_count, 40)
        self.assertGreater(calibrated.issue_lift_max, calibrated.issue_lift_min)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        reference = torch.full((1, 3, 8, 8), 0.2)
        candidate = torch.full((1, 3, 8, 8), 0.22)
        self.certification = Certifier().certify(
            reference, candidate, torch.ones(1, 1, 8, 8), safe_inputs(), profile()
        )
        self.uncertainty = StructuredUncertainty(
            source_supported=True,
            residual_interval_width=0.08,
            diagnosis_reliable=True,
            teacher_agreement=True,
            projection_retention=0.9,
            arbiter_available=True,
            arbiter_order_stable=True,
            arbiter_preference="PROJECTED",
            metadata_complete=True,
            ovis_available=False,
            ovis_hard_defect=False,
        )

    def candidate(self, **overrides):
        values = dict(
            teacher_level="L3",
            raw_generated=False,
            projected=True,
            full_recertified=True,
            certification=self.certification,
            uncertainty=self.uncertainty,
            parameter_stable=True,
            range_bounded=True,
            real_data=False,
            calibration_profile=profile(),
        )
        values.update(overrides)
        return SupervisionCandidate(**values)

    def test_structured_router_uses_necessary_conditions_not_additive_score(self):
        router = SupervisionRouter(pixel_route_enabled=True)
        self.assertFalse(hasattr(self.uncertainty, "score"))
        self.assertIs(router.route(self.candidate(teacher_level="L1")).route, Route.PARAMETER)
        self.assertIs(router.route(self.candidate(teacher_level="L2")).route, Route.RANGE)
        self.assertIs(router.route(self.candidate()).route, Route.PIXEL)

    def test_raw_injection_and_uncalibrated_real_pixel_fail_closed(self):
        router = SupervisionRouter(pixel_route_enabled=True)
        raw = router.route(self.candidate(raw_generated=True, projected=False, full_recertified=False))
        real_uncalibrated = router.route(
            self.candidate(real_data=True, calibration_profile=profile(synthetic=True))
        )
        unavailable = replace(self.uncertainty, source_supported=False)
        diagnostic = router.route(self.candidate(uncertainty=unavailable))
        self.assertIs(raw.route, Route.REJECT)
        self.assertIs(real_uncalibrated.route, Route.PREFERENCE)
        self.assertIn("real_calibration_profile", real_uncalibrated.reasons)
        self.assertIs(diagnostic.route, Route.DIAGNOSTIC)


class ManifestAndLineageTests(unittest.TestCase):
    def test_complete_hashed_lineage_and_manifest_guards(self):
        root_hash = hash_of("root")
        projected_hash = hash_of("projected")
        store = LineageStore()
        store.add(LineageNode(root_hash, "input", (), ("load",)))
        store.add(LineageNode(projected_hash, "projected_target", (root_hash,), ("tm_projection",)))
        self.assertEqual(store.ancestors(projected_hash), (root_hash,))

        record = ManifestRecord(
            artifact_sha256=projected_hash,
            input_sha256=root_hash,
            parent_sha256s=(root_hash,),
            model_sha256=hash_of("model"),
            config_sha256=hash_of("config"),
            profile_sha256=hash_of("profile"),
            transformations=("tm_projection", "full_recertification"),
            gates=tuple({"name": name, "status": "pass"} for name in CRITICAL_GATE_NAMES),
            supervision_type="pixel",
            synthetic=True,
            real_model=False,
            raw_generated=False,
            projected=True,
            fully_certified=True,
            route_reasons=("synthetic_canary_only",),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            ManifestWriter(path).write(record)
            payload = json.loads(path.read_text().strip())
        self.assertEqual(payload["artifact_sha256"], projected_hash)
        with self.assertRaises(ValueError):
            ManifestRecord.from_mapping({**record.to_dict(), "unknown": 1})
        with self.assertRaises(ValueError):
            ManifestWriter.validate(replace(record, raw_generated=True))


if __name__ == "__main__":
    unittest.main()
