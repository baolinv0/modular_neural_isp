import unittest

import torch
from torch import nn

from cross_camera_tm.adapters import PairTransformParameters, TargetCameraAdapter
from cross_camera_tm.losses import AlignmentQuality, DistillationLoss
from cross_camera_tm.phase1 import (
    FrozenSamsungTM,
    PairParameterSolver,
    Phase1ValidationSample,
    TeacherMetricProfile,
    TeacherQualificationStatus,
    TeacherQualifier,
    assess_phase1_validation,
    fit_pair_parameter_predictor,
    gradient_canary,
)


class TinyTone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, image):
        return {"output": (image * self.scale).clamp(0.0, 1.0)}


class TargetCameraAdapterTests(unittest.TestCase):
    def test_identity_initialization_and_low_capacity(self):
        adapter = TargetCameraAdapter(feature_dim=8, hidden_dim=4)
        image = torch.rand(2, 3, 8, 8) * 0.7
        output = adapter(image, torch.zeros(2, 8), confidence=torch.ones(2))
        self.assertTrue(torch.allclose(output.image, image, atol=1e-6))
        self.assertLessEqual(sum(parameter.numel() for parameter in adapter.parameters()), 256)
        self.assertTrue(torch.allclose(output.curve_y[0], torch.linspace(0, 1, 6), atol=1e-6))

    def test_explicit_order_is_gain_then_matrix_then_curve(self):
        image = torch.tensor([0.2, 0.4, 0.3]).view(1, 3, 1, 1)
        params = PairTransformParameters(
            gains=torch.tensor([[2.0, 0.5, 1.0]]),
            matrix=torch.tensor([[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
            curve_y=torch.linspace(0, 1, 6).view(1, 6),
        )
        output = TargetCameraAdapter.apply_explicit(image, params, confidence=torch.ones(1))
        self.assertTrue(torch.allclose(output, torch.tensor([0.5, 0.2, 0.3]).view(1, 3, 1, 1), atol=1e-6))

    def test_monotonic_curve_and_confidence_gate(self):
        adapter = TargetCameraAdapter(feature_dim=4, hidden_dim=4)
        with torch.no_grad():
            adapter.head.bias.fill_(2.0)
        image = torch.full((1, 3, 4, 4), 0.25)
        active = adapter(image, torch.ones(1, 4), confidence=torch.ones(1))
        disabled = adapter(image, torch.ones(1, 4), confidence=torch.zeros(1))
        self.assertTrue(torch.all(active.curve_y[:, 1:] >= active.curve_y[:, :-1]))
        self.assertTrue(torch.allclose(active.curve_y[:, 0], torch.zeros(1)))
        self.assertTrue(torch.allclose(active.curve_y[:, -1], torch.ones(1)))
        self.assertTrue(torch.allclose(disabled.image, image, atol=1e-6))


class DistillationAndTeacherTests(unittest.TestCase):
    def test_alignment_quality_controls_available_losses(self):
        student = torch.full((1, 3, 8, 8), 0.3)
        teacher = student.clone()
        teacher[:, :, 2:6, 2:6] += 0.1
        roi = torch.zeros(1, 1, 8, 8)
        roi[:, :, 2:6, 2:6] = 1
        loss_fn = DistillationLoss()
        scene = loss_fn(student, teacher, alignment=AlignmentQuality.SCENE_ONLY, roi_mask=roi)
        roi_result = loss_fn(student, teacher, alignment=AlignmentQuality.ROI, roi_mask=roi)
        lowfreq = loss_fn(
            student,
            teacher,
            alignment=AlignmentQuality.LOW_FREQUENCY,
            roi_mask=roi,
            alignment_mask=torch.ones_like(roi),
        )
        self.assertEqual(scene.enabled, ("tone", "adapter"))
        self.assertEqual(roi_result.enabled, ("tone", "roi", "adapter"))
        self.assertEqual(lowfreq.enabled, ("tone", "roi", "lowfreq", "adapter"))
        self.assertEqual(scene.roi.item(), 0.0)
        self.assertGreater(roi_result.roi.item(), 0.0)
        self.assertGreater(lowfreq.lowfreq.item(), 0.0)

    def test_teacher_thresholds_are_calibrated_from_source_distribution(self):
        errors = [
            {"log_luma_rmse": float(value), "highlight_error": float(value) / 2}
            for value in range(1, 21)
        ]
        profile = TeacherMetricProfile.from_source_errors(errors)
        qualifier = TeacherQualifier(profile)
        qualified = qualifier.qualify({"log_luma_rmse": 10.0, "highlight_error": 5.0}, hard_defect=False)
        downweighted = qualifier.qualify({"log_luma_rmse": 17.0, "highlight_error": 8.5}, hard_defect=False)
        rejected = qualifier.qualify({"log_luma_rmse": 20.0, "highlight_error": 10.0}, hard_defect=False)
        self.assertIs(qualified.status, TeacherQualificationStatus.QUALIFIED)
        self.assertIs(downweighted.status, TeacherQualificationStatus.DOWNWEIGHTED)
        self.assertLess(downweighted.weight, 1.0)
        self.assertIs(rejected.status, TeacherQualificationStatus.REJECTED)
        self.assertEqual(qualifier.qualify({"log_luma_rmse": 1.0, "highlight_error": 0.5}, True).weight, 0.0)


class PairSolverAndFrozenTeacherTests(unittest.TestCase):
    def test_pair_initializer_recovers_channel_scale_and_monotonic_curve(self):
        iphone = torch.rand(1, 3, 16, 16) * 0.4 + 0.1
        scale = torch.tensor([1.2, 0.8, 1.05]).view(1, 3, 1, 1)
        samsung = iphone * scale
        solver = PairParameterSolver(curve_points=6)
        initialized = solver.initialize(iphone, samsung, torch.ones(1, 1, 16, 16, dtype=torch.bool))
        self.assertTrue(torch.allclose(initialized.gains, scale.flatten(1), atol=0.03))
        self.assertTrue(torch.all(initialized.curve_y[:, 1:] >= initialized.curve_y[:, :-1]))
        self.assertTrue(torch.allclose(initialized.matrix, torch.eye(3).unsqueeze(0), atol=0.08))

    def test_joint_refinement_reduces_teacher_loss(self):
        tone = FrozenSamsungTM(TinyTone())
        iphone = torch.full((1, 3, 8, 8), 0.2)
        teacher = torch.full((1, 3, 8, 8), 0.36)
        initial = PairTransformParameters.identity(batch=1, device=iphone.device, dtype=iphone.dtype)
        solver = PairParameterSolver(curve_points=6)
        result = solver.refine(
            initial,
            iphone,
            teacher,
            tone,
            DistillationLoss(),
            steps=30,
            learning_rate=0.08,
        )
        self.assertLess(result.final_loss, result.initial_loss)
        self.assertEqual(len(result.loss_history), 31)

    def test_frozen_tm_gradient_canary_requires_input_gradient_only(self):
        tone = FrozenSamsungTM(TinyTone())
        report = gradient_canary(tone, torch.rand(1, 3, 8, 8) * 0.5)
        self.assertTrue(report.output_finite)
        self.assertTrue(report.input_gradient_finite)
        self.assertTrue(report.input_gradient_nonzero)
        self.assertEqual(report.trainable_backbone_parameters, 0)

    def test_pair_parameters_train_tiny_metadata_predictor(self):
        torch.manual_seed(4)
        features = torch.linspace(-1.0, 1.0, 40).view(40, 1).repeat(1, 4)
        targets = PairTransformParameters.identity(
            batch=40, device=features.device, dtype=features.dtype
        )
        targets = PairTransformParameters(
            gains=torch.exp(0.12 * features[:, :1]).repeat(1, 3),
            matrix=targets.matrix,
            curve_y=targets.curve_y,
        )
        adapter = TargetCameraAdapter(feature_dim=4, hidden_dim=4)
        result = fit_pair_parameter_predictor(
            adapter,
            features,
            targets,
            confidence=torch.ones(40),
            steps=120,
            learning_rate=0.03,
        )
        self.assertLess(result.final_loss, result.initial_loss * 0.2)

    def test_phase1_validation_enforces_development_and_locked_direction(self):
        samples = [
            Phase1ValidationSample(
                pair_id=f"pair-{index}",
                scene_group=f"scene-{index % 5}",
                split="development" if index < 40 else "locked",
                baseline_error=0.20,
                adapted_error=0.12 + (index % 3) * 0.005,
                parameter_bound_margin=0.1,
            )
            for index in range(50)
        ]
        result = assess_phase1_validation(samples, bootstrap_samples=200, seed=3)
        self.assertTrue(result.passed)
        self.assertEqual(result.positive_folds, 5)
        self.assertEqual(result.improved_development_pairs, 40)


if __name__ == "__main__":
    unittest.main()
