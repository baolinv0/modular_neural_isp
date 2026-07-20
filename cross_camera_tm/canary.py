from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .adapters import TargetCameraAdapter
from .canonicalization import DeviceCanonicalizer
from .certification import CertificationProfile
from .config import PipelineConfig
from .contracts import FailureType, LinearMetadata
from .phase1 import FrozenSamsungTM
from .pipeline import CrossCameraPipeline
from .residuals import (
    ActivationSample,
    ResidualComponent,
    ResidualEstimate,
    assess_phase2_activation,
)
from .routing import SupervisionRouter


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DeterministicSamsungTMDouble(nn.Module):
    def forward(self, image: torch.Tensor):
        return {"output": (image * 0.98 + 0.005).clamp(0.0, 1.0)}


def _metadata() -> LinearMetadata:
    return LinearMetadata.from_mapping(
        {
            "sample_id": "synthetic-canary-0001",
            "device": "synthetic-iphone-interface-fixture",
            "white_level": 65535.0,
            "is_normalized": False,
            "black_level_corrected": True,
            "white_balanced": True,
            "awb_gains_applied": [1.0, 1.0, 1.0],
            "reference_awb_gains": [1.0, 1.0, 1.0],
            "awb_gains_comparable": True,
            "ccm_to_common": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "exposure_time_s": None,
            "iso": None,
            "aperture": None,
            "reference_exposure_product": None,
            "hdr_confidence": 0.5,
            "metadata_complete": True,
        }
    )


def _profile() -> CertificationProfile:
    return CertificationProfile(
        profile_id="deterministic-synthetic-canary-profile",
        profile_sha256=_sha("deterministic-synthetic-canary-profile-v2"),
        dataset_sha256=_sha("deterministic-synthetic-fixtures-only"),
        synthetic=True,
        sample_count=60,
        issue_lift_min=0.05,
        issue_lift_max=0.18,
        clipping_growth_max=0.01,
        geometry_correlation_min=0.98,
        high_frequency_correlation_min=0.85,
        high_frequency_energy_min=0.70,
        high_frequency_energy_max=1.40,
        chromaticity_mae_max=0.02,
        non_target_correction_max=0.04,
        boundary_artifact_max=0.03,
    )


def run_synthetic_canary(
    *, config: PipelineConfig, output_dir: Path | None
) -> dict[str, object]:
    if config.mode != "synthetic_canary":
        raise ValueError("synthetic canary requires mode=synthetic_canary")
    torch.manual_seed(config.seed)
    base = torch.linspace(0.12, 0.52, 32 * 32, dtype=torch.float32).view(1, 1, 32, 32)
    image = torch.cat((base * 0.98, base, base * 1.02), dim=1) * 65535.0
    adapter = TargetCameraAdapter(feature_dim=8, hidden_dim=4)
    pipeline = CrossCameraPipeline(
        canonicalizer=DeviceCanonicalizer(config.canonicalization),
        target_adapter=adapter,
        samsung_tm=FrozenSamsungTM(DeterministicSamsungTMDouble()),
        router=SupervisionRouter(pixel_route_enabled=config.routing.pixel_route_enabled),
    )
    activation_samples = [
        ActivationSample(
            eligible=True,
            severity=0.12,
            source_p75=0.05,
            scene_group=f"synthetic-scene-{index % 5}",
            source_supported=True,
            phase1_bound_margin=0.2,
            source_replay_regressed=False,
            failure_type=FailureType.GLOBAL_UNDEREXPOSURE,
        )
        for index in range(config.phase2.minimum_eligible_samples + 10)
    ]
    activation = assess_phase2_activation(
        activation_samples, bootstrap_samples=200, seed=config.seed
    )
    residual = ResidualEstimate(
        available=True,
        reason="synthetic_calibrated_fixture",
        global_residual=ResidualComponent(0.10, 0.06, 0.16, 0.02),
        face_residual=ResidualComponent(0.16, 0.10, 0.22, 0.03),
        luma_low=0.08,
        luma_high=0.72,
        support_distance=0.1,
        neighbor_count=5,
    )
    manifest_path = output_dir / "manifest.jsonl" if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.run(
        image,
        _metadata(),
        torch.zeros(1, 8),
        residual,
        _profile(),
        phase2_enabled=config.phase2.enabled,
        phase2_activated=activation.activated,
        failure_type=FailureType.GLOBAL_UNDEREXPOSURE,
        face_mask=None,
        config_sha256=config.sha256,
        model_sha256=_sha("deterministic-samsung-tm-interface-double"),
        synthetic=True,
        real_model=False,
        manifest_path=manifest_path,
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "synthetic": True,
        "real_model": False,
        "real_data_effectiveness_verified": False,
        "fixture": "deterministic_mechanical_canary",
        "seed": config.seed,
        "config_sha256": config.sha256,
        "trace": list(result.trace),
        "route": result.routing.route.value,
        "route_reasons": list(result.routing.reasons),
        "reason": result.reason,
        "phase2_activated": activation.activated,
        "phase2_bootstrap_lower_bound": round(activation.bootstrap_lower_bound, 8),
        "candidate_sha256": result.candidate.image_sha256 if result.candidate is not None else None,
        "certification_accepted": result.certification.accepted if result.certification is not None else False,
        "critical_gate_statuses": {
            gate.name: gate.status.value for gate in result.certification.gates
        }
        if result.certification is not None
        else {},
        "unverified": [
            "50-pair real cross-device calibration effectiveness",
            "real iPhone target holdout improvement",
            "source-domain non-regression on real paired Samsung data",
            "real Qwen3-VL/Qwen-Image-Edit/InternVL/Ovis execution",
        ],
    }
    if output_dir is not None:
        (output_dir / "canary_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report
