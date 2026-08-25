from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portrait_evaluator.config import EvaluatorConfig
from portrait_evaluator.image_utils import load_image
from portrait_evaluator.models import ImageFeatures, ROISet, SceneSpec, ValidationStatus
from portrait_evaluator.roi.engine import FaceDetector, build_rois, face_composition_delta


@dataclass(slots=True)
class ValidatedScene:
    status: ValidationStatus
    reasons: list[str]
    images: dict[str, ImageFeatures]
    rois: dict[str, ROISet]


def validate_scene(spec: SceneSpec, config: EvaluatorConfig, detector: FaceDetector | None = None) -> ValidatedScene:
    reasons: list[str] = []; images: dict[str, ImageFeatures] = {}; rois: dict[str, ROISet] = {}
    for role in ("source", "baseline", "candidate", "reference"):
        path: Path = getattr(spec, role)
        if not path.exists(): return ValidatedScene(ValidationStatus.INVALID, [f"{role} missing: {path}"], {}, {})
        try: images[role] = load_image(path)
        except Exception as exc: return ValidatedScene(ValidationStatus.INVALID, [f"{role} unreadable: {exc}"], {}, {})
        roi, roi_reasons = build_rois(images[role], config, spec.face_bbox.get(role), detector); reasons.extend(f"{role}: {r}" for r in roi_reasons)
        if roi is None: return ValidatedScene(ValidationStatus.INVALID, reasons, images, rois)
        rois[role] = roi
    ref_shape = images["reference"].rgb.shape[:2]; status = ValidationStatus.VALID
    for role in ("candidate", "baseline"):
        center, scale = face_composition_delta(rois[role], rois["reference"], images[role].rgb.shape[:2], ref_shape)
        if center > float(config.get("validation.invalid_face_center_distance")) or scale > float(config.get("validation.invalid_face_scale_ev")):
            return ValidatedScene(ValidationStatus.INVALID, reasons + [f"{role}/reference composition too different: center={center:.3f}, scale_ev={scale:.3f}"], images, rois)
        if center > float(config.get("validation.warn_face_center_distance")) or scale > float(config.get("validation.warn_face_scale_ev")):
            status = ValidationStatus.VALID_WITH_CONFOUNDERS; reasons.append(f"{role}/reference composition confounder: center={center:.3f}, scale_ev={scale:.3f}")
    return ValidatedScene(status, reasons, images, rois)
